#/bin/sh


SRC_FILES = src_f/tps.f95 \
	src_f/rbf.f95 \
	src_f/idw.f95 \
	src_f/utils.f95 \

default:	
	f2py --fcompiler=gnu95 -c ${SRC_FILES} -m lib
	mv lib.so smt
