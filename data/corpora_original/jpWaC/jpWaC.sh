curl --remote-name-all https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1047{/jpWaC-L4.vert.gz,/jpWaC-L3.vert.gz,/jpWaC-L2.vert.gz,/jpWaC-L1.vert.gz,/jpWaC-L0.vert.gz}
# can take some minutes
python process_jpWaC.py --files ./jpWaC-L*.gz