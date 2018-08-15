CC := gcc
CP := g++
LIBFLAGS := -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
SHAREFLAGS := -O3 -shared -fPIC

all: zstates.so linear_regression.so logmarglik.so logmarglik_bslmm.so logmarglik_bslmm_fixedPI.so logmarglik_point_normal.so

zstates.so: zstates_py.cpp
	$(CP) $< $(SHAREFLAGS) -o $@

linear_regression.so: linear_regression.c
	$(CC) $< $(SHAREFLAGS) -o $@

logmarglik.so: logmarglik.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

logmarglik_bslmm.so: logmarglik_bslmm.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

logmarglik_bslmm_fixedPI.so: logmarglik_bslmm_fixedPI.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

logmarglik_point_normal.so: logmarglik_point_normal.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

clean:
	rm -rf zstates.so linear_regression.so logmarglik.so logmarglik_bslmm.so logmarglik_bslmm_fixedPI.so logmarglik_point_normal.so
