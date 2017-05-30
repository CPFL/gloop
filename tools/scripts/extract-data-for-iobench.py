# -*- coding: utf-8 -*-
#  Copyright (C) 2017 Yusuke Suzuki <utatane.tea@gmail.com>
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

import sys
import os
import re
import numpy as np

def main():
    regex = re.compile(r"^(\d+)-(\d+)-(.+): (\d+)$")
    target = sys.argv[1]
    data = {}
    with open(target) as file:
        for line in file:
            matched = regex.match(line)
            ioSize = int(matched.group(1))
            loopCount = int(matched.group(2))
            kind = matched.group(3)
            value = int(matched.group(4))
            if not data.has_key(ioSize):
                data[ioSize] = {}
            layer1 = data[ioSize]
            if not layer1.has_key(loopCount):
                layer1[loopCount] = {}
            layer2 = layer1[loopCount]
            if not layer2.has_key(kind):
                layer2[kind] = []
            layer3 = layer2[kind]
            layer3.append(value)

        for kind in ['normal', 'double']:
            print kind
            shown = False
            for ioSize in sorted(data):
                result = []
                result.append(str(ioSize))
                loopCounts = sorted(data[ioSize])
                for loopCount in loopCounts:
                    array = data[ioSize][loopCount][kind]
                    array.pop(0)
                    array = np.array(array)
                    result.append("%f" % (np.mean(array)))
                    # print ioSize, loopCount, np.mean(array)
                if not shown:
                    print "0\t%s" % ("\t".join([ str(v) for v in loopCounts ]))
                    shown = True
                print "\t".join(result)

if __name__ == '__main__':
    main()
# vim: set sw=4 ts=4 et tw=80 :
