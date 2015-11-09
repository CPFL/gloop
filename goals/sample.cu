/*
  Copyright (C) 2015 Yusuke Suzuki <yusuke.suzuki@sslab.ics.keio.ac.jp>

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
  ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
  THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// The goal of this project.
// Type 1.
__global__ void run2()
{
    gloopOpen("...", [&] (file) {
        gloopRead(file, [&] (data) {
            calcluate(data);
            gloopWrite(data, [&] (int err) {
                // process errors.
            });
        });
    });
}

// Type 2.
__global__ void run2()
{
    gloopOpen("...").flatMap([&] (file) {
        return gloopRead(file).flatMap([&] (data) {
            calcluate(data);
            return gloopWrite(data).flatMap([&] (int err) {
                // process errors.
            });
        });
    });
}

// Then, Type 3. Introducing do-syntax.
__global__ gloop::Promise<...> run2()
{
    do {
        file <- gloopOpen("...");
        data <- gloopRead(file);
        calculate(data);
        gloopWrite(data);
        return data;
    }
}

int main(int argc, char** argv)
{
    run<<>>();
    return 0;
}
