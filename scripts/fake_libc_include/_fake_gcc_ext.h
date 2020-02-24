#ifndef _FAKE_GCC_EXT_H
#define _FAKE_GCC_EXT_H

/* Header file to remove (some) GCC extensions */

/*
Get rid of the `__asm__(...)` & `__asm__ volatile(...)` syntax. This does not break volatile type qualifiers.
However, this leaves `__asm__ volatile(...)` to be a single `__asm__`.
Thus, we typedef `__asm__` before the define, so it becomes a single type, which is a valid statement.
*/
typedef int __asm__;
typedef int __asm;
#define __asm__(...)
#define __asm(...)
#define volatile(...)

#define __attribute__(...)
#define __attribute(...)

#define __const__ const
#define __const const
#define __inline__ inline
#define __inline inline
#define __restrict__ restrict
#define __restrict restrict
#define __volatile__ volatile
#define __volatile volatile
#define __extension__

#endif // _FAKE_GCC_EXT_H
