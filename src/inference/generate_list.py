import os
def generate_text(folder, fake_types):
    text = ''
    for fake_type in os.listdir(folder):
        if (os.path.isdir(os.path.join(folder, fake_type))):
            for video in os.listdir(os.path.join(folder, fake_type)):
                label = 0
                if fake_type in fake_types:
                    label = 1
                text += f'{label} {os.path.join(fake_type, video)}\n'
    return text

def save_text(path, text_title, text):
    with open(os.path.join(path, text_title), "w") as file:
        file.write(text)

if __name__ == '__main__':
    FOLDER = '/media/alicia/T7/ShareID/TestDataSets/alexandre_master'
    FAKE_TYPES = [folder for folder in os.listdir(FOLDER) if os.path.isdir(os.path.join(FOLDER, folder))]
    print(f"Fake types: {FAKE_TYPES}")
    text = generate_text(FOLDER, FAKE_TYPES)
    save_text(FOLDER, 'List_of_testing_videos.txt', text)