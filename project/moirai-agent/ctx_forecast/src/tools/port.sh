host_ip=ip-10-3-1-223

port_moirai_v2=1111
port_chronos=9999
ssh hsyan@$host_ip -L $port_moirai_v2:localhost:$port_moirai_v2 -L $port_chronos:localhost:$port_chronos

port_ipython=11111
ssh hsyan@ip-10-3-98-125 -L $port_ipython:localhost:$port_ipython


# # Show all listening ports
# netstat -tuln

# # Find unused ports in a specific range (e.g., 8000-8010)
# for port in {9800..9999}; do
#     netstat -tuln | grep ":$port " > /dev/null || echo "Port $port is available"
# done
