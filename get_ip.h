#ifndef GET_IP_H
#define GET_IP_H
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <ifaddrs.h>

#define MAX_IFACE_NAME 16

void get_ip(char **ip) {
    struct ifaddrs *ifap, *ifa;
    struct sockaddr_in *sa;
    
    if (getifaddrs(&ifap) == -1) {
        perror("getifaddrs");
        exit(EXIT_FAILURE);
    }

    for (ifa = ifap; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr != NULL && ifa->ifa_addr->sa_family == AF_INET) {
            sa = (struct sockaddr_in *) ifa->ifa_addr;
            
            // Skip loopback interfaces
            if (strcmp(ifa->ifa_name, "lo") != 0) {
                *ip = strdup(inet_ntoa(sa->sin_addr));
                break;
            }
        }
    }

    freeifaddrs(ifap);
}
#endif