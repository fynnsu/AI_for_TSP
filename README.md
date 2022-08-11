# AI_for_TSP

This Repository contains the winning solution to the AI For TSP Competition Track 2 (https://github.com/paulorocosta/ai-for-tsp-competition).

# Important Scripts
- generate_train_instances.py - creates a dataset of training instances
- Train.py - trains a base POMO model on a dataset of instances
- active_search_instance.py - performs efficient active search (EAS) on a single test instance, creating improved node embeddings
- create_tour_instance.py - uses EAS node embeddings, creates a tour for a single test instance, using MC tree searches to improve performance
- combine_tours.py - combines tours from multiple instances into the required submission format

# Team Members
- Fynn Schmitt-Ulms
- André Hottung
- Kevin Tierney
- Meinolf Sellmann

# Acknowledgements
Our solution is built using code from the original POMO paper by Kwon et al. \
Paper: https://arxiv.org/abs/2010.16011 \
Code: https://github.com/yd-kwon/POMO

We extend our solution using the Efficient Active Search method described by Hottung et al. \
Paper: https://arxiv.org/abs/2106.05126 \
Code: https://github.com/ahottung/EAS
