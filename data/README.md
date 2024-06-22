## Synthetic test data
The python generator can be used to create some synthetic data for testing.

## Caida data sets
Additional files are available [DHS](https://github.com/ZeBraHack0/DHS/tree/main/data). 

These originated from [CAIDA-2016](http://www.caida.org/data/passive/passive_2016_dataset.xml) and [CAIDA-2019](http://www.caida.org/data/passive/passive_2019_dataset.xml).
If you want to use CAIDA datasets, please register [CAIDA](http://www.caida.org/home/) and then apply for the traces.

## Test Data Format

These data files are binary files in big-endian. 
Each 13-byte string is a network five-tuple in the format of (srcIP, dstIP, srcPort, dstPort, protocol). 
For a 13-byte string "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c", the ASCII code should be:

- srcIP = "\x00\x01\x02\x03"
- dstIP = "\x04\x05\x06\x07"
- srcPort = "\x08\x09"
- dstPort = "\x0a\x0b"
- protocol = "\x0c"
