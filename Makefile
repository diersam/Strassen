CXX=icpc
#CXX=g++
CPPFLAGS= -Ofast -march=native -Wall -Wconversion -Wshadow -Wnon-virtual-dtor -std=c++17 -fPIC -fopenmp
LD=$(CXX)
LDLIBS=-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
LDFLAGS=
SRC := $(wildcard *.cpp)
OBJS = $(SRC:.cpp=.o)

all: $(OBJS)

test: all
	$(LD) -o test $(OBJS) $(LDLIBS) $(LDFLAGS) 

clean:
	rm -f $(OBJS)
	rm -f *.a *.so

ctags:
	ctags -R --c++-kinds=+p --fields=+iaS --extra=+q
