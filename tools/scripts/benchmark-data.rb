# -*- coding: utf-8 -*-
#  Copyright (C) 2016 Yusuke Suzuki <utatane.tea@gmail.com>
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
#  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

require_relative './gloop.rb'

module BenchmarkData
    module TPACF
        Data = <<-EOS.gsub('\n', '').strip
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Datapnts.1,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.1,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.2,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.3,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.4,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.5,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.6,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.7,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.8,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.9,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.10,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.11,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.12,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.13,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.14,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.15,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.16,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.17,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.18,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.19,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.20,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.21,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.22,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.23,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.24,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.25,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.26,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.27,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.28,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.29,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.30,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.31,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.32,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.33,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.34,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.35,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.36,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.37,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.38,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.39,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.40,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.41,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.42,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.43,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.44,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.45,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.46,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.47,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.48,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.49,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.50,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.51,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.52,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.53,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.54,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.55,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.56,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.57,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.58,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.59,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.60,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.61,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.62,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.63,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.64,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.65,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.66,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.67,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.68,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.69,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.70,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.71,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.72,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.73,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.74,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.75,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.76,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.77,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.78,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.79,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.80,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.81,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.82,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.83,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.84,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.85,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.86,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.87,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.88,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.89,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.90,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.91,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.92,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.93,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.94,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.95,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.96,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.97,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.98,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.99,
/home/yusukesuzuki/dev/parboil2-gloop/parboil/datasets/tpacf/large/input/Randompnts.100
        EOS
    end
end

# vim: set sw=4 ts=4 et tw=80 :
