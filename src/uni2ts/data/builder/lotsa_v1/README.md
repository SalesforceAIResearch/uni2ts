### ProEnFo

1. Download the relevant data files from the [ProEnFo repository](https://github.com/Leo-VK/ProEnFo), and place them in folders with their respective dataset names.
2. Register the directory path with the ProEnFo datasets by running: ```echo "PROENFO_PATH=ADD_YOUR_PATH" >> .env```
3. Run the following command to process the datasets: ```python -m uni2ts.builder.lotsa_v1 proenfo```

`
Warning! The command above reads the pickle data files provided by the ProEnFo repository.
Deserializing pickle files can lead to remote code execution.
Please be careful when dealing with untrusted pickle files.
`