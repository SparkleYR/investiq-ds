# InvestIQ

This is a basic graph-based project that models Indian deeptech startup funding as an investor-startup bipartite network and recommends likely future investor-startup links using engineered graph features and a Random Forest model.

- Loads and clean the Indian Startup Funding dataset locally(https://www.kaggle.com/code/sudalairajkumar/simple-exploration-notebook-indian-startups) - for now.
- Filters likely deeptech startups using keyword heuristics.
- Builds startup, investor, and transaction entity modules.
- Creates a bipartite NetworkX graph and an investor co-investment projection.
- Computes simple ecosystem metrics such as degree centrality, PageRank, and communities.
- Trains a baseline link-prediction model for investor recommendations.
- Includes a small Streamlit app for ecosystem viewing and startup-to-investor matching. 

## Future scope

to be written soon.