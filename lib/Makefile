CC := gcc
CP := g++
LIBFLAGS := -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
# SHAREFLAGS := -O3 -shared -fPIC
SHAREFLAGS := -O3 -shared -fPIC

all: logmarglik_original.so logmarglik.so logmarglik_bslmm.so zstates.so linear_regression.so

logmarglik_original.so: logmarglik_original.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

logmarglik.so: logmarglik.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

logmarglik_bslmm.so: logmarglik_bslmm.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

zstates.so: zstates_py.cpp
	$(CP) $< $(SHAREFLAGS) -o $@

linear_regression.so: linear_regression.c
	$(CC) $< $(SHAREFLAGS) -o $@

clean:
	rm -rf zstates.so logmarglik.so linear_regression.so logmarglik_bslmm.so logmarglik_original.so
