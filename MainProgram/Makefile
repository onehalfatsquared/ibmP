# set up compiler and options
CXX = g++
CC = gcc
CXXFLAGS = -O3 -fopenmp -march=native $(INCLUDES) 

#-----Suffix Rules---------------------------
# set up C++ suffixes and relationship between .cpp and .o files
.SUFFIXES: .cpp

%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

#-----File Dependencies----------------------

SRC = 	IBMain.cpp ImmersedFiber.cpp 

OBJ = $(addsuffix .o, $(basename $(SRC)))

IBMain: $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJ)
clean:
		rm -rf *.o *~ DTProgress *.index *.txt *.out
