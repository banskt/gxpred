CC := gcc -O3
LIBFLAGS := -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_sequential.a ${MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm -ldl
SHAREFLAGS := -shared -fPIC

#trial: trial.o
#	$(CC) $< $(LIBFLAGS) -o $@
#trial.o: trial.cpp
#	$(CC) -c $< -o $@
#

margloglik.so: margloglik.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

trial.so: trial.c
	$(CC) $< $(LIBFLAGS) $(SHAREFLAGS) -o $@

clean:
	rm -rf margloglik.so
