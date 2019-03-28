# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
sys.path.append("../common")

from builtins import range
from future.utils import iteritems
import unittest
import numpy as np
import infer_util as iu
import test_util as tu
from tensorrtserver.api import *
import os

np_dtype_string = np.dtype(object)

class InferZeroTest(unittest.TestCase):

    def _full_zero(self, dtype, shapes):
        # 'shapes' is list of shapes, one for each input.

        # For validation assume any shape can be used...
        if tu.validate_for_tf_model(dtype, dtype, dtype, shapes[0], shapes[0], shapes[0]):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_zero(self, 'graphdef', bs, dtype, shapes, shapes)
                iu.infer_zero(self, 'savedmodel', bs, dtype, shapes, shapes)
            # model that does not support batching
            iu.infer_zero(self, 'graphdef_nobatch', 1, dtype, shapes, shapes)
            iu.infer_zero(self, 'savedmodel_nobatch', 1, dtype, shapes, shapes)

        if tu.validate_for_c2_model(dtype, dtype, dtype, shapes[0], shapes[0], shapes[0]):
            # model that supports batching
            for bs in (1, 8):
                iu.infer_zero(self, 'netdef', bs, dtype, shapes, shapes)
            # model that does not support batching
            iu.infer_zero(self, 'netdef_nobatch', 1, dtype, shapes, shapes)

    def test_ff1_sanity(self):
        self._full_zero(np.float32, ([1,],))
    def test_ff1(self):
        self._full_zero(np.float32, ([0,],))
    def test_ff3_sanity(self):
        self._full_zero(np.float32, ([1,],[2,],[1,]))
    def test_ff3_0(self):
        self._full_zero(np.float32, ([0,],[0,],[0,]))
    def test_ff3_1(self):
        self._full_zero(np.float32, ([0,],[0,],[1,]))
    def test_ff3_2(self):
        self._full_zero(np.float32, ([0,],[1,],[0,]))
    def test_ff3_3(self):
        self._full_zero(np.float32, ([1,],[0,],[0,]))
    def test_ff3_4(self):
        self._full_zero(np.float32, ([1,],[0,],[1,]))

    def test_hh1_sanity(self):
        self._full_zero(np.float16, ([2, 2],))
    def test_hh1_0(self):
        self._full_zero(np.float16, ([1, 0],))
    def test_hh1_1(self):
        self._full_zero(np.float16, ([0, 1],))
    def test_hh1_2(self):
        self._full_zero(np.float16, ([0, 0],))

    def test_hh3_sanity(self):
        self._full_zero(np.float16, ([2, 2],[2, 2],[1, 1]))
    def test_hh3_0(self):
        self._full_zero(np.float16, ([0, 0],[0, 0],[0, 0]))
    def test_hh3_1(self):
        self._full_zero(np.float16, ([0, 1],[0, 1],[2,3]))
    def test_hh3_2(self):
        self._full_zero(np.float16, ([1, 0],[1, 3],[0, 1]))
    def test_hh3_3(self):
        self._full_zero(np.float16, ([1, 1],[3, 0],[0, 0]))
    def test_hh3_4(self):
        self._full_zero(np.float16, ([1, 1],[0, 6],[2, 2]))

    def test_oo1_sanity(self):
        self._full_zero(np_dtype_string, ([2,],))
    def test_oo1(self):
        self._full_zero(np_dtype_string, ([0,],))

    def test_oo3_sanity(self):
        self._full_zero(np_dtype_string, ([2, 2],[2, 2],[1, 1]))
    def test_oo3_0(self):
        self._full_zero(np_dtype_string, ([0, 0],[0, 0],[0, 0]))
    def test_oo3_1(self):
        self._full_zero(np_dtype_string, ([0, 1],[0, 1],[2,3]))
    def test_oo3_2(self):
        self._full_zero(np_dtype_string, ([1, 0],[1, 3],[0, 1]))
    def test_oo3_3(self):
        self._full_zero(np_dtype_string, ([1, 1],[3, 0],[0, 0]))
    def test_oo3_4(self):
        self._full_zero(np_dtype_string, ([1, 1],[0, 6],[2, 2]))


if __name__ == '__main__':
    unittest.main()