CC := gcc
LIBFLAGS := -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
SHAREFLAGS := -O3 -shared -fPIC

ll: logmarglik.so zstates.so linear_regression.so

logmarglik.so: logmarglik.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

zstates.so: zstates_py.cpp
	$(CC) $< $(SHAREFLAGS) -o $@

linear_regression.so: linear_regression.c
	$(CC) $< $(SHAREFLAGS) -o $@

clean:
	rm -rf zstates.so logmarglik.so linear_regression.so
