# python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir ~/data/sim_transfer_cube_scripted --num_episodes 50 --onscreen_render
# if args less than 2, return
if [ $# -lt 2 ]; then
    echo "Usage: $0 dataset_dir episode_idx"
    exit 1
fi

# if exist third parameter, use it to decide run visualize_episodes or visualize_episodes_real
if [ $# -eq 3 ]; then
    if [ $3 == "real" ]; then
        python3 aloha_scripts/visualize_episodes.py  --dataset_dir ~/data/aloha/$1 --episode_idx $2
        exit 1
    fi
fi
python3 visualize_episodes.py  --dataset_dir ~/data/aloha/$1 --episode_idx $2
