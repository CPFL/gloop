#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

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

static float& a(std::vector<float>& vec, int i, int j)
{
    return vec[MATRIX_HW * i + j];
}

static float& b(std::vector<float>& vec, int i, int j)
{
    return vec[MATRIX_SIZE + (MATRIX_HW * i + j)];
}

static float& c(std::vector<float>& vec, int i, int j)
{
    return a(vec, i, j);
}

int main(int argc, char *argv[])
{
	int sock;
	// struct sockaddr_in server;

	if (argc < 3) {
		microbench_usage_client(argc, argv);
		exit(1);
	}

	sock = microbench_client_connect(argv[1], argv[2]);

	puts("Connected\n");
    {
        std::vector<float> vec(MATRIX_SIZE * 2, 1.0f);
        std::vector<float> result(MATRIX_SIZE, 0);
        int res = 0;
        res = send(sock, (void*)vec.data(), vec.size() * sizeof(float), 0);
        printf("send OK %d\n", res);
        res = recv(sock, (void*)result.data(), result.size() * sizeof(float), MSG_WAITALL);
        printf("recv OK %d\n", res);

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
    }
	// bench_send_recv_bw<MSG_SIZE, NR_MSG>(sock);

	close(sock);
	return 0;
}
