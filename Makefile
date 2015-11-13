CC=g++
LIBS=-Lcnn/build/cnn -lcnn -lboost_serialization -lboost_filesystem -lboost_system -lboost_program_options -lstdc++ -lm
CFLAGS=-Icnn -Icnn/eigen -I./cnn/external/easyloggingpp/src -std=gnu++11 -g
OBJ=util.o training.o main-dclm.o baseline.o dam.o

all: main-dclm baseline dam

%.o: %.cc
	$(CC) $(CFLAGS) -c -o $@ $< 

main-dclm: main-dclm.o training.o test.o sample.o util.o dclm-output.hpp dclm-hidden.hpp rnnlm.hpp
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

baseline: baseline.o util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

dam: dam.o util.o
	$(CC) $(CFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -rf *.o *.*~ main-dclm baseline dam

