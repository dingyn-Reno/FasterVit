include ../../include/Makefile.inc

SOURCE_CPP  = $(shell find . -name '*.cpp' 2>/dev/null)
SOURCE_PY   = $(shell find . -name 'main.py' 2>/dev/null)
OBJ         = $(shell find . -name *.o 2>/dev/null)
DEP         = $(OBJ:.o=.d)
TARGET_EXE  = $(SOURCE_CPP:.cpp=.exe)

-include $(DEP)

all: $(TARGET_EXE)

%.exe: %.o
	$(NVCC) $(CCFLAG) $(LDFLAG) -o $@ $+

%.o: %.cpp
	$(NVCC) $(CCFLAG) $(INCLUDE) -M -MT $@ -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAG) $(INCLUDE) -Xcompiler -fPIC -o $@ -c $<

.PHONY: test
test:
	make clean
	make
	python3 $(SOURCE_PY)
	nsys profile -o async -f true python3 $(SOURCE_PY)
	rm -rf ./*.plan
	./$(TARGET_EXE)

.PHONY: clean
clean:
	rm -rf ./*.d ./*.o ./*.so ./*.exe ./*.plan
