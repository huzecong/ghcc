import abc
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Set, Type

import pymongo
from mypy_extensions import TypedDict

__all__ = [
    "RepoDB",
    "BinaryDB",
    "MatchFuncDB",
]

Index = Dict[str, Any]


class BaseEntry(TypedDict, total=False):
    r"""The base class for MongoDB entries. Setting ``total=False`` allows not setting the ``_id`` key when creating a
    dictionary to be inserted.
    """
    _id: Any  # actually `bson.ObjectId`, but we don't care


class Database(abc.ABC):
    r"""A wrapper over MongoDB that handles the connection and authentication. This is an abstract base class, concrete
    classes must override the :meth:`collection_name` property.
    """

    # For some reason, mypy thinks `TypedDict` is a function...
    Entry: Type[TypedDict]  # type: ignore

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
    def index(self) -> List[Index]:
        r"""Sets of key(s) to use as index. Each key is represented as a tuple of (name, order), where order is
        ``pymongo.ASCENDING`` (1) or ``pymongo.DESCENDING`` (-1).

        By default, all indexes are unique. To add a non-unique index, include ``"$unique": False`` in the dictionary.
        """
        return []

    def __init__(self, config_file: str = "./database-config.json", **override_config):
        r"""Create a connection to the database.
        """
        if not os.path.exists(config_file):
            raise ValueError(f"DB config file not found at '{config_file}'. "
                             f"Please refer to 'database-config-example.json' for the format")
        with open(config_file) as f:
            config: Database.Config = json.load(f)
        config.update(override_config)
        missing_keys = [key for key in Database.Config.__annotations__ if key not in config]
        if len(missing_keys) > 0:
            raise ValueError(f"Keys {missing_keys} are missing from the DB config file at '{config_file}'.from "
                             f"Please refer to 'database-config-example.json' for the format")

        self.client = pymongo.MongoClient(
            config['host'], port=config['port'], authSource=config['auth_db_name'],
            username=config['username'], password=config['password'])
        self.collection = self.client[config['db_name']][self.collection_name]

        for new_index in self.index:
            new_index = new_index.copy()
            unique = True
            if "$unique" in new_index:
                unique = new_index["$unique"]
                del new_index["$unique"]
            for key in new_index:
                if key not in self.Entry.__annotations__:
                    raise ValueError(f"Index contains key '{key}', which is not in Entry definition")
            if not any(index["key"].to_dict() == new_index for index in self.collection.list_indexes()):
                # Only create index if no such index exists.
                # This check is required because `create_index` seems not idempotent, although it should be.
                self.collection.create_index(list(new_index.items()), unique=unique, background=True)

    def count(self, estimate: bool = True) -> int:
        if estimate:
            return self.collection.estimated_document_count()
        return self.collection.count_documents({})

    def close(self) -> None:
        del self.collection
        self.client.close()

    def safe_iter(self, batch_size: int = 1000, static: bool = False) -> Iterator['Entry']:  # type: ignore
        r"""Safely iterate over all documents. The normal way (``for entry in collection.find()``) result in cursor
        timeout if the iteration takes too long, and it's unavoidable unless configured on the server.

        The approach here follows that in the StackOverflow answer (https://stackoverflow.com/a/40434043/4909228). We
        choose a unique index, sort the entire collection according to the index, and then fetch a batch of entries.
        When the batch is depleted, a new batch is fetched.

        :param batch_size: The size (number of entries) of each fetched batch. A larger batch sizer reduces iteration
            overhead, but uses more memory. Defaults to 1000.
        :param static: Whether the DB is static, i.e., whether there would be modifications during iteration. If not,
            a set of IDs for iterated entries will be recorded to prevent yielding the same entry twice, which incurs
            overhead.
        :return: An iterator over all entries.
        """
        # Find a unique index.
        if all(not index.get("$unique", True) for index in self.index):
            raise ValueError(f"`safe_iter` does not work for database {self.__class__.__name__} because there are no "
                             f"unique indexes.")
        index = next(index for index in self.index if index.get("$unique", True))
        if "$unique" in index:
            del index["$unique"]

        yielded_ids: Set[Any] = set()
        prev_index = 0
        while True:
            cursor = self.collection.find().sort(list(index.items())).skip(prev_index).limit(batch_size)
            if static:
                entries = list(cursor)
            else:
                entries = []
                for entry in cursor:
                    if entry["_id"] not in yielded_ids:
                        entries.append(entry)
                        yielded_ids.add(entry["_id"])
            if len(entries) == 0:
                break
            yield from entries
            prev_index += batch_size


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
    def index(self) -> List[Index]:
        return [{
            "repo_owner": pymongo.ASCENDING,
            "repo_name": pymongo.ASCENDING,
        }]

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
        repo_owner: str
        repo_name: str
        sha: str
        success: bool

    @property
    def collection_name(self) -> str:
        return "binaries"

    @property
    def index(self) -> List[Index]:
        return [
            {"repo_owner": pymongo.ASCENDING,
             "repo_name": pymongo.ASCENDING,
             "$unique": False},
            {"sha": pymongo.ASCENDING},
        ]

    def get(self, sha: str) -> Optional[Entry]:
        r"""Get the DB entry corresponding to the specified binary hash.

        :return: If entry exists, it is returned as a dictionary; otherwise ``None`` is returned.
        """
        return self.collection.find_one({"sha": sha})

    def get_binaries_by_repo(self, repo_owner: str, repo_name: str, success: bool = True) -> Iterator[Entry]:
        r"""Get all matching DB entries given a repository.

        :param repo_owner: Owner of the repository.
        :param repo_name: Name of the repository.
        :param success: The decompilation status of the binary.
        :return: An iterator (``pymongo.Cursor``) over matching entries.
        """
        return self.collection.find({"repo_owner": repo_owner, "repo_name": repo_name, "success": success})

    def add_binary(self, repo_owner: str, repo_name: str, sha: str, success: bool) -> None:
        r"""Add a new DB entry for the specified binary.

        :param repo_owner: Owner of the repository.
        :param repo_name: Name of the repository.
        :param sha: Hash of the binary.
        :param success: Whether the binary was successfully decompiled.
        """
        record = self.get(sha)
        if record is None:
            record = {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "sha": sha,
                "success": success,
            }
            self.collection.insert_one(record)
        else:
            self.collection.update_one({"_id": record["_id"]}, {"$set": {
                "success": success,
            }})


class MatchFuncDB(Database):
    class Entry(BaseEntry):
        repo_owner: str
        repo_name: str
        files_found: int
        funcs_found: int
        funcs_matched: int
        funcs_matched_without_ast: int

    @property
    def collection_name(self) -> str:
        return "match_func"

    @property
    def index(self) -> List[Index]:
        return [{
            "repo_owner": pymongo.ASCENDING,
            "repo_name": pymongo.ASCENDING,
        }]

    def get(self, repo_owner: str, repo_name: str) -> Optional[Entry]:
        r"""Get the DB entry corresponding to the specified repository.

        :return: If entry exists, it is returned as a dictionary; otherwise ``None`` is returned.
        """
        return self.collection.find_one({"repo_owner": repo_owner, "repo_name": repo_name})

    def add_repo(self, repo_owner: str, repo_name: str,
                 files_found: int, funcs_found: int, funcs_matched: int, funcs_matched_without_ast: int) -> None:
        r"""Add a new DB entry for the specified repository."""
        record = self.get(repo_owner, repo_name)
        if record is None:
            record = {
                "repo_owner": repo_owner,
                "repo_name": repo_name,
                "files_found": files_found,
                "funcs_found": funcs_found,
                "funcs_matched": funcs_matched,
                "funcs_matched_without_ast": funcs_matched_without_ast,
            }
            self.collection.insert_one(record)
        else:
            self.collection.update_one({"_id": record["_id"]}, {"$set": {
                "files_found": files_found,
                "funcs_found": funcs_found,
                "funcs_matched": funcs_matched,
                "funcs_matched_without_ast": funcs_matched_without_ast,
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
