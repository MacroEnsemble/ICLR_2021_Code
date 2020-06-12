#!/bin/bash
SYSLOAD=`awk '{print "f={"$1"}f={"$2"}f={"$3"}"}' /proc/loadavg`
CPUINFO=`lscpu | grep -E '^Thread|^Core|^Socket'`

THREAD=`echo "${CPUINFO}" | egrep '^Thread' | sed 's/^[^:]*:[ \t]*\([0-9]\+\).*$/\1/'`
CORE=`echo "${CPUINFO}" | egrep '^Core'   | sed 's/^[^:]*:[ \t]*\([0-9]\+\).*$/\1/'`
SOCKET=`echo "${CPUINFO}" | egrep '^Socket' | sed 's/^[^:]*:[ \t]*\([0-9]\+\).*$/\1/'`
TOTAL=`echo "$THREAD $CORE $SOCKET" | awk '{print $1*$2*$3}'`
echo "${SYSLOAD}i={${THREAD}}i={${CORE}}i={${SOCKET}}i={${TOTAL}}"

