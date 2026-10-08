#include <iostream>
#include <iomanip>
#include <cmath>

using namespace std;

long double cal(long int n, long int i, long int j) {
    long double num = (long double)n + i - j;
    long double den = (long double)n * n +
                      (long double)n * i -
                      ((long double)n * (n + 1)) / 2.0L;
    return num / den;
}

int main() {

    const int N = 10;
    const long double EPS = 1e-18L;

    cout << fixed << setprecision(30);

    for (int i1 = 1; i1 <= N; i1++) {
        for (int i2 = i1 + 1; i2 <= N; i2++) {
            for (int k = 1; k <= N; k++) {

                long double temp1 = 0.0L;
                long double temp2 = 0.0L;

                for (int j = k; j <= N; j++) {
                    temp1 += cal(N, i1, j);
                    temp2 += cal(N, i2, j);
                }

                // Compare with tolerance
                if (temp1 > temp2 + EPS) {
                    cout << "Found violation:\n";
                    cout << "i1 = " << i1
                         << ", i2 = " << i2
                         << ", k = " << k << '\n';

                    cout << "temp1 = " << temp1 << '\n';
                    cout << "temp2 = " << temp2 << '\n';
                    cout << "Difference = " << temp1 - temp2 << '\n';

                    return 0;
                }
            }
        }
    }

    cout << "done\n";
    return 0;
}