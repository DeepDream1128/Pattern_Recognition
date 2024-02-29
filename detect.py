import openalpr_api

alpr = openalpr_api.Alpr("us", "/etc/openalpr/openalpr.conf", "/usr/share/openalpr/runtime_data")
if not alpr.is_loaded():
    print("Error loading OpenALPR")
    sys.exit(1)

results = alpr.recognize_file("/path/to/image.jpg")

print(results)

alpr.unload()
