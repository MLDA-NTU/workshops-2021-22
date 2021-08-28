import os

# Replace with the appropriate folder name
folder = "RGB"
files = os.listdir(folder)

# Gets all the XML files into a list
xml_files = [x for x in files if x[-4:]==".xml"] 

# print(xml_files)

# Write paths.txt, with each line being the XML file path
with open('paths.txt', 'w') as filehandle:
    for listitem in xml_files:
        filehandle.write(f'./{folder}/%s\n' % listitem)