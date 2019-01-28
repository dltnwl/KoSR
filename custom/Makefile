CC=g++
CFLAGS=-std=c++11 -shared -fPIC -O2
TF_FLAGS=$(shell python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags() + tf.sysconfig.get_link_flags()))')

.PHONY: all clean

all: jamo_search.so

%.so: %.cpp %.h
	$(CC) $(CFLAGS) -o $@ $< ${TF_FLAGS}

clean:
	rm -rf *.so
