/*
CheMPS2: a spin-adapted implementation of DMRG for ab initio quantum chemistry
Copyright (C) 2013-2018 Sebastian Wouters

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/
#pragma once

double wigner3j(const int two_ja, const int two_jb, const int two_jc,
                const int two_ma, const int two_mb, const int two_mc);

double wigner6j(const int two_ja, const int two_jb, const int two_jc, 
                const int two_jd, const int two_je, const int two_jf);

double wigner9j(const int two_ja, const int two_jb, const int two_jc,
                const int two_jd, const int two_je, const int two_jf,
                const int two_jg, const int two_jh, const int two_ji);
