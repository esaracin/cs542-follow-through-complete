from UserModel import *
import pickle

def top_3_from_file(mp4, pro_dict):
    u = UserModel()
    u.add_sample(mp4)
    u_name = mp4.split('/')[-1][:-5]
    distance_list = []

    print(u_name + '\'s jumpshot', 'is closest to...')
    for pro in pro_dict:
        pro_name = pro.split('/')[-1][:-4]
        
        dist = np.linalg.norm(u.get_vector() - pro_dict[pro].get_vector())
        distance_list.append([dist, pro_name]) 

    sorted_distances = sorted(distance_list)
    for i in range(3):
        print(sorted_distances[i])

    print()
    

def main():
    with open('test_files/pro_user_dict.pickle', 'rb') as f:
        pro_dict = pickle.load(f)

#    pro_users = list(pro_dict.keys())
    for pro in pro_dict:
        pro_name = pro.split('/')[-1][:-4]
        print(pro_name, 'is closest to...')
        distance_list = []
        for other_pro in pro_dict:
            other_pro_name = other_pro.split('/')[-1][:-4]
            if pro == other_pro:
                continue

            dist = np.linalg.norm(pro_dict[pro].get_vector() - pro_dict[other_pro].get_vector())
            distance_list.append([dist, other_pro_name]) 

        sorted_distances = sorted(distance_list)
        for i in range(3):
            print(sorted_distances[i])

        print()


    top_3_from_file('input_mp4s/2kvids/jordan3.mp4', pro_dict)

if __name__ == '__main__':
    main()
