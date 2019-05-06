from UserModel import *
import os

def main():
    user_dict = {}
    files_added = []

    if os.path.isfile('test_files/pro_users_seen.pickle'):
        with open('test_files/pro_users_seen.pickle', 'rb') as f:
            files_added = pickle.load(f) 

    if os.path.isfile('test_files/pro_user_dict.pickle'):
        with open('test_files/pro_user_dict.pickle', 'rb') as f:
            user_dict = pickle.load(f) 

    for f in glob.iglob('./input_mp4s/2kvids/*'):
        if f in files_added:
            continue

        print(f)
        files_added.append(f)

        athlete = f[:-5] + f[-4:]
        if athlete in user_dict:
            u = user_dict[athlete]
            u.add_sample(f)
        else:
            u = UserModel()
            u.add_sample(f)
            user_dict[athlete] = u

        with open('./test_files/pro_user_dict.pickle', 'wb') as handle:
            pickle.dump(user_dict, handle)

        with open('./test_files/pro_users_seen.pickle', 'wb') as handle:
            pickle.dump(files_added, handle)


if __name__ == '__main__':
    main()
