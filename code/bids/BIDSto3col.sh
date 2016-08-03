#!/bin/bash
#
# Script: BIDSto3col.sh
# Purpose: Convert a BIDS event TSV file to a 3 column FSL file
# Author: T Nichols t.e.nichols@warwick.ac.uk
# Version: 1.1   17 Feb 2016
#

# Extension to give created 3 column files
ThreeColExt="txt"

# To avoid headaches with spaces in filenames, replace spaces in event names 
# with this character.
SpaceReplace="_"


###############################################################################
#
# Environment set up
#
###############################################################################

shopt -s nullglob # No-match globbing expands to null
Tmp=/tmp/`basename $0`-${$}-
trap CleanUp INT

###############################################################################
#
# Functions
#
###############################################################################

Usage() {
cat <<EOF
Usage: `basename $0` [options] BidsTSV OutBase 

Reads BidsTSV and then creates 3 column event files, one per event type, 
using the basename OutBase.  By default, all event types are used, and the 
height value (3rd column) is 1.0.

Options
  -e EventName   Instead of all event types, only use the given event type.
  -h ColName     Instead of using 1.0, get height value from given column.
  -N             By default, when creating 3 column files any spaces in the 
                 event name are replaced with "$SpaceReplace"; use this option to
                 prevent this replacement.
EOF
exit
}

CleanUp () {
    /bin/rm -f /tmp/`basename $0`-${$}-*
    exit 0
}


###############################################################################
#
# Parse arguments
#
###############################################################################

while (( $# > 1 )) ; do
    case "$1" in
        "-help")
            Usage
            ;;
        "-e")
            shift
            EventNm="$1"
            shift
            ;;
        "-h")
            shift
            HeightNm="$1"
            shift
            ;;
        "-N")
            shift
            NoSpaceRepl=1
            ;;
        -*)
            echo "ERROR: Unknown option '$1'"
            exit 1
            break
            ;;
        *)
            break
            ;;
    esac
done

if (( $# < 2 )) ; then
    Usage
fi

TSV="$1"
OutBase="$2"

###############################################################################
#
# Script Body
#
###############################################################################

# Validate TSV
if [ ! -f "$TSV" ] ; then
    echo "ERROR: Cannot find '$TSV'."
    CleanUp
fi

# Get all event names  (need to loop to handle spaces)
awk -F'\t' '(NR>1){print $3}' "$TSV" | sort | uniq > ${Tmp}AllEv
nEV=$(cat ${Tmp}AllEv | wc -l)
EventNms=()
for ((i=1;i<=nEV;i++)); do 
    EventNms[i-1]="$(sed -n ${i}p ${Tmp}AllEv)"
done

# Validate requested event name
if [ "$EventNm" != "" ] ; then
    Fail=1
    for ((i=0;i<${#EventNms[*]};i++)) ; do
	if [ "${EventNms[i]}" = "$EventNm" ] ; then
	    Fail=0
	    break
	fi
    done
    if [ $Fail = 1 ] ; then
	echo "ERROR: Event type '$EventNm' not found in TSV file."
	CleanUp
    fi
    EventNms=("$EventNm")
fi

# Validate height column name
if [ "$HeightNm" != "" ] ; then
    VarNms=( $(awk -F'\t' '(NR==1){print $0}' "$TSV") )
    for ((i=2;i<${#VarNms[*]};i++)) ; do
	if [ "${VarNms[i]}" = "$HeightNm" ] ; then
	    ((HeightCol=i+1));
	    break
	fi
    done
    if [ "$HeightCol" = "" ] ; then
	echo "ERROR: Column '$HeightNm' not found in TSV file."
	CleanUp
    fi
fi


for E in "${EventNms[@]}" ; do

    if [ "$NoSpaceRepl" = 1 ] ; then
	Out="$OutBase"_"$E"."$ThreeColExt"
    else
	Out="$OutBase"_"$(echo $E | sed 's/ /'$SpaceReplace'/g')"."$ThreeColExt"
    fi

    #echo "Creating '$Out'... "

    if [ "$HeightNm" = "" ] ; then

	awk -F'\t' '$3~/^'"$E"'$/{printf("%s	%s	1.0\n",$1,$2)}'                "$TSV" > "$Out"

    else

	awk -F'\t' '$3~/^'"$E"'$/{printf("%s	%s	%s\n",$1,$2,$'"$HeightCol"')}' "$TSV" > "$Out"

	# Validate height values
	Nev="$(cat "$Out" | wc -l)"
	Nnum="$(awk '{print $3}' "$Out" | grep -E '^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$' | wc -l )"
	if [ "$Nev" != "$Nnum" ] ; then
	    echo "	WARNING: Event '$E' has non-numeric heights from '$HeightNm'"
	fi

    fi

done


###############################################################################
#
# Exit & Clean up
#
###############################################################################

CleanUp
