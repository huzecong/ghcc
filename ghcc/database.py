import abc
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pymongo
from mypy_extensions import TypedDict

__all__ = [
    "RepoDB",
    "BinaryDB",
]


class BaseEntry(TypedDict, total=False):
    r"""The base class for MongoDB entries. Setting ``total=False`` allows not setting the ``_id`` key when creating a
    dictionary to be inserted.
    """
    _id: Any  # actually `bson.ObjectId`, but we don't care


class Database(abc.ABC):
    r"""A wrapper over MongoDB that handles the connection and authentication. This is an abstract base class, concrete
    classes must override the :meth:`collection_name` property.
    """

    class Config(TypedDict):
        host: str
        port: int
        auth_db_name: str
        db_name: str
        username: str
        password: str

    @property
    @abc.abstractmethod
    def collection_name(self) -> str:
        r"""Name of the collection that the DB object uses."""
        raise NotImplementedError

    @property
    def index(self) -> Dict[str, int]:
        r"""Key(s) to use as index. Each key is represented as a tuple of (name, order), where order is
        ``pymongo.ASCENDING`` (1) or ``pymongo.DESCENDING`` (-1)."""
        return {}

    def __init__(self, config_file: str = "./database-config.json"):
        r"""Create a connection to the database.
        """
        if not os.path.exists(config_file):
            raise ValueError(f"DB config file not found at '{config_file}'. "
                             f"Please refer to 'database-config-example.json' for the format")
        with open(config_file) as f:
            config: Database.Config = json.load(f)
        missing_keys = [key for key in Database.Config.__annotations__ if key not in config]
        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} are missing from the DB config file at '{config_file}'.from "
                             f"Please refer to 'database-config-example.json' for the format")

        self.client = pymongo.MongoClient(
            config['host'], port=config['port'], authSource=config['auth_db_name'],
            username=config['username'], password=config['password'])
        self.collection = self.client[config['db_name']][self.collection_name]

        if len(self.index) > 0:
            if not any(index["key"].to_dict() == self.index for index in self.collection.list_indexes()):
                # Only create index if no such index exists.
                # This check is required because `create_index` seems not idempotent, although it should be.
                self.collection.create_index(list(self.index.items()), unique=True, background=True)

    def close(self) -> None:
        del self.collection
        self.client.close()


class RepoDB(Database):
    r"""An abstraction over MongoDB that stores information about repositories.
    """

    class MakefileEntry(BaseEntry):
        directory: str  # directory containing the Makefile
        success: bool  # whether compilation was successful (return code 0)
        binaries: List[str]  # list of paths to binaries generated by make operation
        sha256: List[str]  # SHA256 hashes for each binary

    class Entry(BaseEntry):
        repo_owner: str
        repo_name: str
        repo_size: int  # size of the repo in bytes
        clone_successful: bool  # whether the repo has been successfully cloned to the server
        compiled: bool  # whether the repo has been tested for compilation
        num_makefiles: int  # number of compilable Makefiles (required because MongoDB cannot aggregate list lengths)
        num_binaries: int  # number of generated binaries (required because MongoDB cannot aggregate list lengths)
        makefiles: 'List[RepoDB.MakefileEntry]'  # list of Makefiles

    @property
    def collection_name(self) -> str:
        return "repos"

    @property
    def index(self) -> Dict[str, int]:
        return {
            "repo_owner": pymongo.ASCENDING,
            "repo_name": pymongo.ASCENDING,
        }

    def get(self, repo_owner: str, repo_name: str) -> Optional[Entry]:
        r"""Get the DB entry corresponding to the specified repository.

        :return: If entry exists, it is returned as a dictionary; otherwise ``None`` is returned.
        """
        return self.collection.find_one({"repo_owner": repo_owner, "repo_name": repo_name})

    def add_repo(self, repo_owner: str, repo_name: str, clone_successful: bool, repo_size: int = -1) -> None:
        r"""Add a new DB entry for the specified repository. Arguments correspond to the first three fields in
        :class:`RepoEntry`. Other fields are set to sensible default values (``False`` and ``[]``).

        :param repo_owner: Owner of the repository.
        :param repo_name: Name of the repository.
        :param clone_successful: Whether the repository was successfully cloned.
        :param repo_size: Size (in bytes) of the cloned repository, or ``-1`` (default) if cloning failed.
        :return: The internal ID of the inserted entry.
        """
        record = self.get(repo_owner, repo_name)
        if record is None:
            record = {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "clone_successful": clone_successful,
                "repo_size": repo_size,
                "compiled": False,
                "num_makefiles": 0,
                "num_binaries": 0,
                "makefiles": [],
            }
            self.collection.insert_one(record)
        else:
            self.collection.update_one({"_id": record["_id"]}, {"$set": {
                "clone_successful": clone_successful,
                "repo_size": repo_size,
            }})

    def update_makefile(self, repo_owner: str, repo_name: str, makefiles: List[MakefileEntry],
                        ignore_length_mismatch: bool = False) -> bool:
        r"""Update Makefile compilation results for a given repository.

        :param repo_owner: Owner of the repository.
        :param repo_name: Name of the repository.
        :param makefiles: List of Makefile compilation results.
        :param ignore_length_mismatch: If ``False``, a :exc:`ValueError` is raised if the number of Makefiles previously
            stored in the DB is different from the length of :attr:`makefiles` (unless there were no Makefiles
            previously).
        :return: A boolean value, indicating whether the write succeeded. Note that it is considered unsuccessful if
            the :attr:`makefiles` list was not stored due to Unicode encoding errors.
        """
        entry = self.get(repo_owner, repo_name)
        if entry is None:
            raise ValueError(f"Specified repository {repo_owner}/{repo_name} does not exist")
        if not ignore_length_mismatch and len(entry["makefiles"]) not in [0, len(makefiles)]:
            raise ValueError(f"Number of makefiles stored in entry of {repo_owner}/{repo_name} "
                             f"({len(entry['makefiles'])}) does not match provided list ({len(makefiles)})")
        update_entries = {
            "compiled": True,
            "num_makefiles": len(makefiles),
            "num_binaries": sum(len(makefile["binaries"]) for makefile in makefiles),
            "makefiles": makefiles,
        }
        try:
            result = self.collection.update_one({"_id": entry["_id"]}, {"$set": update_entries})
            assert result.matched_count == 1
            return True
        except UnicodeEncodeError as e:
            update_entries["makefiles"] = []  # some path might contain strange characters; just don't store it
            result = self.collection.update_one({"_id": entry["_id"]}, {"$set": update_entries})
            assert result.matched_count == 1
            return False

    def _aggregate_sum(self, field_name: str) -> int:
        cursor = self.collection.aggregate(
            [{"$match": {"compiled": True}},
             {"$group": {"_id": None, "total": {"$sum": f"${field_name}"}}}])
        return next(cursor)["total"]

    def count_makefiles(self) -> int:
        return self._aggregate_sum("num_makefiles")

    def count_binaries(self) -> int:
        return self._aggregate_sum("num_binaries")


class BinaryDB(Database):
    class Entry(BaseEntry):
        sha: str
        success: bool

    @property
    def collection_name(self) -> str:
        return "binaries"

    @property
    def index(self) -> Dict[str, int]:
        return {"sha": pymongo.ASCENDING}

    def get(self, sha: str) -> Optional[Entry]:
        r"""Get the DB entry corresponding to the specified binary hash.

        :return: If entry exists, it is returned as a dictionary; otherwise ``None`` is returned.
        """
        return self.collection.find_one({"sha": sha})

    def add_binary(self, sha: str, success: bool) -> None:
        r"""Add a new DB entry for the specified binary.

        :param sha: Hash of the binary.
        :param success: Whether the binary was successfully decompiled.
        :return: The internal ID of the inserted entry.
        """
        record = self.get(sha)
        if record is None:
            record = {
                "sha": sha,
                "success": success,
            }
            self.collection.insert_one(record)
        else:
            self.collection.update_one({"_id": record["_id"]}, {"$set": {
                "success": success,
            }})


if __name__ == '__main__':
    db = RepoDB()
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        confirm = input("This will drop the entire repository database. Confirm? [y/N] ")
        if confirm.lower() in ["y", "yes"]:
            db.collection.delete_many({})
            db.close()
            print("Database dropped.")
        else:
            print("Operation cancelled.")
    else:
        print("Interact with the database using `db`, e.g.:\n"
              "> db.count_makefiles()\n")
        from IPython import embed

        embed()
