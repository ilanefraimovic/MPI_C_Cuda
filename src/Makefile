CC = gcc
CFLAGS = -Wall -Wextra -g

all: BFS

BFS: main.o
	$(CC) $(CFLAGS) -o BFS main.o

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f BFS *.o

