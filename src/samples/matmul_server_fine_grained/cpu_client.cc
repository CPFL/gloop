#include <arpa/inet.h>
#include <boost/thread.hpp>
#include <memory>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <gloop/benchmark.h>

#include "microbench_util_cpu.h"
#include "matmul_server_config.h"

int count_same(char* str, int len) {
	char key = str[0];
	int i;
	for (i = 0; i < len; i++) {
		if (str[i] != key)
			break;
	}
	return i;
}

float& a(std::vector<float>& vec, int i, int j)
{
    return vec[MATRIX_HW * i + j];
}

float& b(std::vector<float>& vec, int i, int j)
{
    return vec[MATRIX_SIZE + (MATRIX_HW * i + j)];
}

float& c(std::vector<float>& vec, int i, int j)
{
    return a(vec, i, j);
}

int main(int argc, char *argv[])
{
	// struct sockaddr_in server;

	if (argc < 3) {
		microbench_usage_client(argc, argv);
		exit(1);
	}

    int nthreads = BLOCKS;
    if (argc >= 4)
        nthreads = atoi(argv[3]);

    int ntimes = 10;
    if (argc >= 5)
        ntimes = atoi(argv[4]);

    int waiting = 0;
    if (argc >= 6)
        waiting = atoi(argv[5]);

    printf("%d\n", waiting);
    {
        std::vector<std::shared_ptr<boost::thread>> threads;
        for (int i = 0; i < nthreads; ++i) {
            threads.push_back(std::make_shared<boost::thread>([=] {
                std::vector<float> vec(MATRIX_SIZE * 2, 1.0f);
                std::vector<float> result(MATRIX_SIZE, 0);
                int res = 0;
                int sock = microbench_client_connect(argv[1], argv[2]);
                gloop::Benchmark benchmark;
                benchmark.begin();
                for (int j = 0; j < ntimes; ++j) {
                    // printf("[%d][%d]\n", i, j);
                    int res = 0;
                    res = send(sock, (void*)vec.data(), vec.size() * sizeof(float), 0);
                    if (res < 0)
                        std::abort();

                    res = recv(sock, (void*)result.data(), result.size() * sizeof(float), MSG_WAITALL);
                    if (res < 0)
                        std::abort();

                    if (waiting) {
                        struct timespec value {
                            0,
                            1000 * waiting // us
                        };
                        struct timespec rem {
                            0,
                            0
                        };
                        while (nanosleep(&value, &rem) == -1) {
                            value = rem;
                            rem = { 0, 0 };
                        }
                    }
                }
                res = close(sock);
                if (res < 0)
                    std::abort();

                benchmark.end();
                benchmark.report(stderr, "times: ");
#if 0
                for (int i = 0; i < MATRIX_HW; ++i) {
                    for (int j = 0; j < MATRIX_HW; ++j) {
                        float actualResult = c(result, i, j);
                        float result = 0.0f;
                        for (int k = 0; k < MATRIX_HW; ++k) {
                            result += a(vec, i, k) * b(vec, k, j);
                        }
                        printf("%f / %f\n", actualResult, result);
                    }
                }
#endif
            }));
        }
        for (auto thread : threads) {
            thread->join();
        }
    }
	// bench_send_recv_bw<MSG_SIZE, NR_MSG>(sock);

	return 0;
}
