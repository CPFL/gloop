# -*- coding: utf-8 -*-
#  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>
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

import os
import sys
from clang.cindex import CursorKind
import clang.cindex

class API(object):
    def __init__(self, node):
        self.node = node

    def report(self):
        result_type = self.node.type.get_result()

        args = []
        for arg in self.node.get_arguments():
            args.append("%s %s" % (arg.type.spelling, arg.spelling))

        print "%s %s(%s);" % (result_type.spelling, self.node.spelling, ", ".join(args))

    def generate_redirector_method(self):
        template = """
%s %s(%s)
{
    return this->m_%s(%s);
}
"""
        result_type = self.node.type.get_result()

        args = []
        for arg in self.node.get_arguments():
            args.append("%s %s" % (arg.type.spelling, arg.spelling))

        return template % (
                result_type.spelling,
                self.node.spelling,
                ", ".join(args),
                self.node.spelling,
                ", ".join([arg.spelling for arg in self.node.get_arguments()]))

    def generate_redirector_member(self):
        template = """
typedef %s (*API%s)(%s);
API%s m_%s;
"""
        result_type = self.node.type.get_result()
        return template % (
                result_type.spelling,
                self.node.spelling,
                ", ".join([arg.type.spelling for arg in self.node.get_arguments()]),
                self.node.spelling,
                self.node.spelling)

    def generate_dlsym(self):
        template = """    m_%s = reinterpret_cast<API%s>(dlsym(RTLD_NEXT, "%s"));
"""
        return template % (
                self.node.spelling,
                self.node.spelling,
                self.node.spelling)


class Generator(object):
    def __init__(self):
        self.api = []

    def visit(self, node, parent):
        self.perform(node, parent)
        for child in node.get_children():
            self.visit(child, node)

    def perform(self, node, parent):
        if node.kind != CursorKind.FUNCTION_DECL:
            return
        self.api.append(API(node))

    def generate_redirector_method(self):
        header = "public:"
        sys.stdout.write(header)
        for api in self.api:
            sys.stdout.write(api.generate_redirector_method())

        members = "private:"
        sys.stdout.write(members)
        for api in self.api:
            sys.stdout.write(api.generate_redirector_member())

        footer = """
Redirector()
{
"""
        sys.stdout.write(footer)
        for api in self.api:
            sys.stdout.write(api.generate_dlsym())
        sys.stdout.write("}\n")

def main():
    index = clang.cindex.Index.create()
    translation_unit = index.parse("../../third_party/cuda-5.0/cuda_runtime_api.h")
    gen = Generator()
    gen.visit(translation_unit.cursor, None)
    if sys.argv[1] == "redirector-method":
        gen.generate_redirector_method()

if __name__ == '__main__':
    main()

# vim: set sw=4 et ts=4:
