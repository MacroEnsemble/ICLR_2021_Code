#!/bin/bash
top -bn2 | grep "Cpu(s)" | \
           sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | \
           tail -n1 | \
           awk '{print "f={"100 - $1"}"}'
