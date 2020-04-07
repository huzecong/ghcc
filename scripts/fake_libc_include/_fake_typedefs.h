#ifndef _FAKE_TYPEDEFS_H
#define _FAKE_TYPEDEFS_H

/* Automatically generated from glibc 2.26 */

typedef int __builtin_va_list;
typedef unsigned char __u_char;
typedef unsigned short int __u_short;
typedef unsigned int __u_int;
typedef unsigned long int __u_long;
typedef signed char __int8_t;
typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned short int __uint16_t;
typedef signed int __int32_t;
typedef unsigned int __uint32_t;
typedef signed long int __int64_t;
typedef unsigned long int __uint64_t;
typedef long int __quad_t;
typedef unsigned long int __u_quad_t;
typedef unsigned long int __dev_t;
typedef unsigned int __uid_t;
typedef unsigned int __gid_t;
typedef unsigned long int __ino_t;
typedef unsigned long int __ino64_t;
typedef unsigned int __mode_t;
typedef unsigned long int __nlink_t;
typedef long int __off_t;
typedef long int __off64_t;
typedef int __pid_t;
typedef struct
{
  int __val[2];
} __fsid_t;
typedef long int __clock_t;
typedef unsigned long int __rlim_t;
typedef unsigned long int __rlim64_t;
typedef unsigned int __id_t;
typedef long int __time_t;
typedef unsigned int __useconds_t;
typedef long int __suseconds_t;
typedef int __daddr_t;
typedef int __key_t;
typedef int __clockid_t;
typedef void *__timer_t;
typedef long int __blksize_t;
typedef long int __blkcnt_t;
typedef long int __blkcnt64_t;
typedef unsigned long int __fsblkcnt_t;
typedef unsigned long int __fsblkcnt64_t;
typedef unsigned long int __fsfilcnt_t;
typedef unsigned long int __fsfilcnt64_t;
typedef long int __fsword_t;
typedef long int __ssize_t;
typedef long int __syscall_slong_t;
typedef unsigned long int __syscall_ulong_t;
typedef __off64_t __loff_t;
typedef __quad_t *__qaddr_t;
typedef char *__caddr_t;
typedef long int __intptr_t;
typedef unsigned int __socklen_t;
typedef __u_char u_char;
typedef __u_short u_short;
typedef __u_int u_int;
typedef __u_long u_long;
typedef __quad_t quad_t;
typedef __u_quad_t u_quad_t;
typedef __fsid_t fsid_t;
typedef __loff_t loff_t;
typedef __ino_t ino_t;
typedef __dev_t dev_t;
typedef __gid_t gid_t;
typedef __mode_t mode_t;
typedef __nlink_t nlink_t;
typedef __uid_t uid_t;
typedef __off_t off_t;
typedef __pid_t pid_t;
typedef __id_t id_t;
typedef __ssize_t ssize_t;
typedef __daddr_t daddr_t;
typedef __caddr_t caddr_t;
typedef __key_t key_t;
typedef __clock_t clock_t;
typedef __time_t time_t;
typedef __clockid_t clockid_t;
typedef __timer_t timer_t;
typedef long unsigned int size_t;
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
typedef int int8_t;
typedef int int16_t;
typedef int int32_t;
typedef int int64_t;
typedef unsigned int u_int8_t;
typedef unsigned int u_int16_t;
typedef unsigned int u_int32_t;
typedef unsigned int u_int64_t;
typedef int register_t;
typedef int __sig_atomic_t;
typedef struct
{
  unsigned long int __val[1024 / (8 * (sizeof(unsigned long int)))];
} __sigset_t;
typedef __sigset_t sigset_t;
typedef __suseconds_t suseconds_t;
typedef long int __fd_mask;
typedef struct
{
  __fd_mask __fds_bits[1024 / (8 * ((int) (sizeof(__fd_mask))))];
} fd_set;
typedef __fd_mask fd_mask;
typedef __blksize_t blksize_t;
typedef __blkcnt_t blkcnt_t;
typedef __fsblkcnt_t fsblkcnt_t;
typedef __fsfilcnt_t fsfilcnt_t;
typedef unsigned long int pthread_t;
typedef union pthread_attr_t pthread_attr_t;
typedef struct __pthread_internal_list
{
  struct __pthread_internal_list *__prev;
  struct __pthread_internal_list *__next;
} __pthread_list_t;
typedef union
{
  struct __pthread_mutex_s
  {
    int __lock;
    unsigned int __count;
    int __owner;
    unsigned int __nusers;
    int __kind;
    short __spins;
    short __elision;
    __pthread_list_t __list;
  } __data;
  char __size[40];
  long int __align;
} pthread_mutex_t;
typedef union
{
  char __size[4];
  int __align;
} pthread_mutexattr_t;
typedef union
{
  struct
  {
    int __lock;
    unsigned int __futex;
    unsigned long long int __total_seq;
    unsigned long long int __wakeup_seq;
    unsigned long long int __woken_seq;
    void *__mutex;
    unsigned int __nwaiters;
    unsigned int __broadcast_seq;
  } __data;
  char __size[48];
  long long int __align;
} pthread_cond_t;
typedef union
{
  char __size[4];
  int __align;
} pthread_condattr_t;
typedef unsigned int pthread_key_t;
typedef int pthread_once_t;
typedef union
{
  struct
  {
    int __lock;
    unsigned int __nr_readers;
    unsigned int __readers_wakeup;
    unsigned int __writer_wakeup;
    unsigned int __nr_readers_queued;
    unsigned int __nr_writers_queued;
    int __writer;
    int __shared;
    signed char __rwelision;
    unsigned char __pad1[7];
    unsigned long int __pad2;
    unsigned int __flags;
  } __data;
  char __size[56];
  long int __align;
} pthread_rwlock_t;
typedef union
{
  char __size[8];
  long int __align;
} pthread_rwlockattr_t;
typedef volatile int pthread_spinlock_t;
typedef union
{
  char __size[32];
  long int __align;
} pthread_barrier_t;
typedef union
{
  char __size[4];
  int __align;
} pthread_barrierattr_t;
typedef union sigval
{
  int sival_int;
  void *sival_ptr;
} sigval_t;
typedef struct sigevent
{
  sigval_t sigev_value;
  int sigev_signo;
  int sigev_notify;
  union
  {
    int _pad[(64 / (sizeof(int))) - 4];
    __pid_t _tid;
    struct
    {
      void (*_function)(sigval_t);
      pthread_attr_t *_attribute;
    } _sigev_thread;
  } _sigev_un;
} sigevent_t;
typedef unsigned char uint8_t;
typedef unsigned short int uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long int uint64_t;
typedef signed char int_least8_t;
typedef short int int_least16_t;
typedef int int_least32_t;
typedef long int int_least64_t;
typedef unsigned char uint_least8_t;
typedef unsigned short int uint_least16_t;
typedef unsigned int uint_least32_t;
typedef unsigned long int uint_least64_t;
typedef signed char int_fast8_t;
typedef long int int_fast16_t;
typedef long int int_fast32_t;
typedef long int int_fast64_t;
typedef unsigned char uint_fast8_t;
typedef unsigned long int uint_fast16_t;
typedef unsigned long int uint_fast32_t;
typedef unsigned long int uint_fast64_t;
typedef long int intptr_t;
typedef unsigned long int uintptr_t;
typedef long int intmax_t;
typedef unsigned long int uintmax_t;
typedef __socklen_t socklen_t;
typedef unsigned short int sa_family_t;
typedef uint32_t in_addr_t;
typedef uint16_t in_port_t;
typedef struct __locale_struct
{
  struct __locale_data *__locales[13];
  const unsigned short int *__ctype_b;
  const int *__ctype_tolower;
  const int *__ctype_toupper;
  const char *__names[13];
} *__locale_t;
typedef __locale_t locale_t;
typedef struct __dirstream DIR;
typedef unsigned short int fexcept_t;
typedef struct
{
  unsigned short int __control_word;
  unsigned short int __glibc_reserved1;
  unsigned short int __status_word;
  unsigned short int __glibc_reserved2;
  unsigned short int __tags;
  unsigned short int __glibc_reserved3;
  unsigned int __eip;
  unsigned short int __cs_selector;
  unsigned int __opcode : 11;
  unsigned int __glibc_reserved4 : 5;
  unsigned int __data_offset;
  unsigned short int __data_selector;
  unsigned short int __glibc_reserved5;
  unsigned int __mxcsr;
} fenv_t;
typedef int (*__ftw_func_t)(const char *__filename, const struct stat *__status, int __flag);
typedef struct
{
  size_t gl_pathc;
  char **gl_pathv;
  size_t gl_offs;
  int gl_flags;
  void (*gl_closedir)(void *);
  void *(*gl_readdir)(void *);
  void *(*gl_opendir)(const char *);
  int (*gl_lstat)(const char *, void *);
  int (*gl_stat)(const char *, void *);
} glob_t;
typedef struct _IO_FILE FILE;
typedef void *iconv_t;
typedef int __gwchar_t;
typedef struct
{
  long int quot;
  long int rem;
} imaxdiv_t;
typedef void *nl_catd;
typedef int nl_item;
typedef float float_t;
typedef double double_t;
typedef enum
{
  _IEEE_ = -1,
  _SVID_,
  _XOPEN_,
  _POSIX_,
  _ISOC_
} _LIB_VERSION_TYPE;
typedef int mqd_t;
typedef u_int32_t tcp_seq;
typedef unsigned long int nfds_t;
typedef unsigned long int __cpu_mask;
typedef struct
{
  __cpu_mask __bits[1024 / (8 * (sizeof(__cpu_mask)))];
} cpu_set_t;
typedef long int __jmp_buf[8];
typedef struct
{
  struct
  {
    __jmp_buf __cancel_jmp_buf;
    int __mask_was_saved;
  } __cancel_jmp_buf[1];
  void *__pad[4];
} __pthread_unwind_buf_t;
typedef long int s_reg_t;
typedef unsigned long int active_reg_t;
typedef unsigned long int reg_syntax_t;
typedef enum
{
  REG_ENOSYS = -1,
  REG_NOERROR = 0,
  REG_NOMATCH,
  REG_BADPAT,
  REG_ECOLLATE,
  REG_ECTYPE,
  REG_EESCAPE,
  REG_ESUBREG,
  REG_EBRACK,
  REG_EPAREN,
  REG_EBRACE,
  REG_BADBR,
  REG_ERANGE,
  REG_ESPACE,
  REG_BADRPT,
  REG_EEND,
  REG_ESIZE,
  REG_ERPAREN
} reg_errcode_t;
typedef struct re_pattern_buffer regex_t;
typedef int regoff_t;
typedef struct
{
  regoff_t rm_so;
  regoff_t rm_eo;
} regmatch_t;
typedef int (*__compar_fn_t)(const void *, const void *);
typedef enum
{
  FIND,
  ENTER
} ACTION;
typedef struct entry
{
  char *key;
  void *data;
} ENTRY;
typedef enum
{
  preorder,
  postorder,
  endorder,
  leaf
} VISIT;
typedef void (*__action_fn_t)(const void *__nodep, VISIT __value, int __level);
typedef union
{
  char __size[32];
  long int __align;
} sem_t;
typedef struct __jmp_buf_tag jmp_buf[1];
typedef struct __jmp_buf_tag sigjmp_buf[1];
typedef __sig_atomic_t sig_atomic_t;
typedef __clock_t __sigchld_clock_t;
typedef struct
{
  int si_signo;
  int si_errno;
  int si_code;
  union
  {
    int _pad[(128 / (sizeof(int))) - 4];
    struct
    {
      __pid_t si_pid;
      __uid_t si_uid;
    } _kill;
    struct
    {
      int si_tid;
      int si_overrun;
      sigval_t si_sigval;
    } _timer;
    struct
    {
      __pid_t si_pid;
      __uid_t si_uid;
      sigval_t si_sigval;
    } _rt;
    struct
    {
      __pid_t si_pid;
      __uid_t si_uid;
      int si_status;
      __sigchld_clock_t si_utime;
      __sigchld_clock_t si_stime;
    } _sigchld;
    struct
    {
      void *si_addr;
      short int si_addr_lsb;
      struct
      {
        void *_lower;
        void *_upper;
      } si_addr_bnd;
    } _sigfault;
    struct
    {
      long int si_band;
      int si_fd;
    } _sigpoll;
    struct
    {
      void *_call_addr;
      int _syscall;
      unsigned int _arch;
    } _sigsys;
  } _sifields;
} siginfo_t;
typedef void (*__sighandler_t)(int);
typedef __sighandler_t sig_t;
typedef struct sigaltstack
{
  void *ss_sp;
  int ss_flags;
  size_t ss_size;
} stack_t;
typedef long long int greg_t;
typedef greg_t gregset_t[23];
typedef struct _libc_fpstate *fpregset_t;
typedef struct
{
  gregset_t gregs;
  fpregset_t fpregs;
  unsigned long long __reserved1[8];
} mcontext_t;
typedef struct ucontext
{
  unsigned long int uc_flags;
  struct ucontext *uc_link;
  stack_t uc_stack;
  mcontext_t uc_mcontext;
  __sigset_t uc_sigmask;
  struct _libc_fpstate __fpregs_mem;
} ucontext_t;
typedef struct
{
  short int __flags;
  pid_t __pgrp;
  sigset_t __sd;
  sigset_t __ss;
  struct sched_param __sp;
  int __policy;
  int __pad[16];
} posix_spawnattr_t;
typedef struct
{
  int __allocated;
  int __used;
  struct __spawn_action *__actions;
  int __pad[16];
} posix_spawn_file_actions_t;
typedef __builtin_va_list __gnuc_va_list;
typedef __gnuc_va_list va_list;
typedef long int ptrdiff_t;
typedef int wchar_t;
typedef struct
{
  long long __max_align_ll;
  long double __max_align_ld;
} max_align_t;
typedef struct _IO_FILE __FILE;
typedef struct
{
  int __count;
  union
  {
    unsigned int __wch;
    char __wchb[4];
  } __value;
} __mbstate_t;
typedef struct
{
  __off_t __pos;
  __mbstate_t __state;
} _G_fpos_t;
typedef struct
{
  __off64_t __pos;
  __mbstate_t __state;
} _G_fpos64_t;
typedef void _IO_lock_t;
typedef struct _IO_FILE _IO_FILE;
typedef __ssize_t __io_read_fn(void *__cookie, char *__buf, size_t __nbytes);
typedef __ssize_t __io_write_fn(void *__cookie, const char *__buf, size_t __n);
typedef int __io_seek_fn(void *__cookie, __off64_t *__pos, int __w);
typedef int __io_close_fn(void *__cookie);
typedef _G_fpos_t fpos_t;
typedef enum
{
  P_ALL,
  P_PID,
  P_PGID
} idtype_t;
typedef union
{
  union wait *__uptr;
  int *__iptr;
} __WAIT_STATUS;
typedef struct
{
  int quot;
  int rem;
} div_t;
typedef struct
{
  long int quot;
  long int rem;
} ldiv_t;
typedef struct
{
  long long int quot;
  long long int rem;
} lldiv_t;
typedef int __t_scalar_t;
typedef unsigned int __t_uscalar_t;
typedef __t_scalar_t t_scalar_t;
typedef __t_uscalar_t t_uscalar_t;
typedef int __ipc_pid_t;
typedef __syscall_ulong_t msgqnum_t;
typedef __syscall_ulong_t msglen_t;
typedef __rlim_t rlim_t;
typedef int __rlimit_resource_t;
typedef int __rusage_who_t;
typedef int __priority_which_t;
typedef __syscall_ulong_t shmatt_t;
typedef struct timezone *__timezone_ptr_t;
typedef int __itimer_which_t;
typedef unsigned char cc_t;
typedef unsigned int speed_t;
typedef unsigned int tcflag_t;
typedef __useconds_t useconds_t;
typedef unsigned int wint_t;
typedef __mbstate_t mbstate_t;
typedef unsigned long int wctype_t;
typedef const __int32_t *wctrans_t;
typedef struct
{
  size_t we_wordc;
  char **we_wordv;
  size_t we_offs;
} wordexp_t;

/* Left-overs from original fake_libc */

typedef int __int_least16_t;
typedef int __uint_least16_t;
typedef int __int_least32_t;
typedef int __uint_least32_t;
typedef int __s8;
typedef int __u8;
typedef int __s16;
typedef int __u16;
typedef int __s32;
typedef int __u32;
typedef int __s64;
typedef int __u64;
typedef int _LOCK_T;
typedef int _LOCK_RECURSIVE_T;
typedef int _flock_t;
typedef int _iconv_t;
typedef int __ULong;
typedef int _types_fd_set;
typedef int cookie_read_function_t;
typedef int cookie_write_function_t;
typedef int cookie_seek_function_t;
typedef int cookie_close_function_t;
typedef int cookie_io_functions_t;
typedef int _sig_func_ptr;
typedef int __tzrule_type;
typedef int __tzinfo_type;
typedef int z_stream;
typedef _Bool bool;
typedef void* MirEGLNativeWindowType;
typedef void* MirEGLNativeDisplayType;
typedef struct MirConnection MirConnection;
typedef struct MirSurface MirSurface;
typedef struct MirSurfaceSpec MirSurfaceSpec;
typedef struct MirScreencast MirScreencast;
typedef struct MirPromptSession MirPromptSession;
typedef struct MirBufferStream MirBufferStream;
typedef struct MirPersistentId MirPersistentId;
typedef struct MirBlob MirBlob;
typedef struct MirDisplayConfig MirDisplayConfig;
typedef struct xcb_connection_t xcb_connection_t;
typedef uint32_t xcb_window_t;
typedef uint32_t xcb_visualid_t;

typedef int __end_of_fake_libc__;

#endif
