#!/bin/bash
#
# Script: BIDSto3col.sh
# Purpose: Convert a BIDS event TSV file to a 3 column FSL file
# Author: T Nichols t.e.nichols@warwick.ac.uk
# Version: 1.2   5 September 2016
#

# Extension to give created 3 column files
ThreeColExt="txt"

# Replace slashes in event names with this character.
SlashReplace=""

# To avoid headaches with spaces in filenames, replace spaces in event names 
# with this character.
SpaceReplace="_"

# Event type column
TypeNm="trial_type"

# Awk header; a command to protect against DOS/Windows carriage returns
AwkHd='{sub(/\r$/,"")};'

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

Reads BidsTSV and then creates 3 column event files, one per event
type if a "$TypeNm" column is found.  Files are named as OutBase
and, if "$TypeNm" is present, appended with the event name. 
By default, all rows and event types are used, and the height value
(3rd column) is 1.0.   

Options
  -s             Even if "$TypeNm" column is found, ignore and process
                 all rows as a single event type.
  -e EventName   Instead of all event types, only use the given event
                 type. 
  -E EventName   Same as -e, except output file name does not have
                 EventName appended.
  -h HtColName   Instead of using 1.0, get height value from given
                 column; two files are written, the unmodulated (with
                 1.0 in 3rd column) and the modulated one, having a
                 "_pmod" suffix. 
  -d DurColName  Instead of getting duration from the "duration"
                 column, take it from this named column.
  -t TypeColName Instead of getting trial type from "trial_type"
                 column, use this column.
  -N             By default, when creating 3 column files any spaces
                 in the event name are replaced with "$SpaceReplace";
                 use this option to suppress this replacement.
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
        "-s")
            shift
            AllEvents=1
            shift
            ;;
        "-e")
            shift
            EventNm="$1"
            shift
            ;;
        "-E")
            shift
            EventNm="$1"
	    NoAppend=1
            shift
            ;;
        "-h")
            shift
            HeightNm="$1"
            shift
            ;;
        "-d")
            shift
            DurNm="$1"
            shift
            ;;
        "-t")
            shift
            TypeNm="$1"
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

# Duration column
if [ "$DurNm" = "" ] ; then
    DurCol=2
else
    # Validate duration column name
    DurCol=$( awk -F'\t' "$AwkHd"'(NR==1){for(i=1;i<=NF;i++){if($i=="'"$DurNm"'")print i}}' "$TSV")
    if [ "$DurCol" = "" ] ; then
	echo "ERROR: Column '$DurNm' not found in TSV file."
	CleanUp
    fi
fi

# Validate trial_type column name
TypeCol=$( awk -F'\t' "$AwkHd"'(NR==1){for(i=1;i<=NF;i++){if($i=="'"$TypeNm"'")print i}}' "$TSV")
if [ "$TypeCol" = "" ] ; then
    echo "WARNING: Column '$TypeNm' not found in TSV file; using all rows."
    AllEvents=1
    NoAppend=1
fi

if [[ "$AllEvents" = "1" ]] ; then

    EventNms=("*")

else

    # Get all event names  (need to loop to handle spaces)
    awk -F'\t' "$AwkHd"'(NR>1){print $'"$TypeCol"'}' "$TSV" | sort | uniq > ${Tmp}AllEv
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
fi

# Validate height column name
if [ "$HeightNm" != "" ] ; then
    HeightCol=$( awk -F'\t' "$AwkHd"'(NR==1){for(i=1;i<=NF;i++){if($i=="'"$HeightNm"'")print i}}' "$TSV")
    if [ "$HeightCol" = "" ] ; then
	echo "ERROR: Column '$HeightNm' not found in TSV file."
	CleanUp
    fi
fi

for E in "${EventNms[@]}" ; do

    if [ "$NoAppend" = 1 ] ; then
	App=""
    else
	E="$(echo "$E" | sed 's@/@'$SlashReplace'@g')"

	if [ "$NoSpaceRepl" = 1 ] ; then
	    App="_${E}"
	else
	    App="_$(echo "$E" | sed 's/ /'$SpaceReplace'/g')"
	fi
    fi

    Out="${OutBase}${App}.${ThreeColExt}"
    OutHt="${OutBase}${App}_pmod.${ThreeColExt}"

    echo "Creating '$Out'... "

    if [ "$AllEvents" = "1" ] ; then
	AwkSel="(NR>1)"
    else
	AwkSel='$'"$TypeCol"'~/^'"$E"'$/'
    fi


    awk -F'\t' "$AwkHd""$AwkSel"'{printf("%s	%s	1.0\n",$1,$'"$DurCol"')}'                "$TSV" > "$Out"

    # Validate duration values (if non-standard duration used)
    if [ "$DurNm" != "" ] ; then
	Nev="$(cat "$Out" | wc -l)"
	Nnum="$(awk '{print $2}' "$Out" | grep -E '^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$' | wc -l )"
	if [ "$Nev" != "$Nnum" ] ; then
            echo "	WARNING: Event '$E' has non-numeric durations from '$DurNm'"
	fi
    fi

    if [ "$HeightNm" != "" ] ; then

	awk -F'\t' "$AwkHd""$AwkSel"'{printf("%s	%s	%s\n",$1,$'"$DurCol"',$'"$HeightCol"')}' "$TSV" > "$OutHt"

	# Validate height values
	Nev="$(cat "$OutHt" | wc -l)"
	Nnum="$(awk '{print $3}' "$OutHt" | grep -E '^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$' | wc -l )"
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
