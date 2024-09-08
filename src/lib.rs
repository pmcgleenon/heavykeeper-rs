//! HeavyKeeper is a library for finding the Top N frequently occuring objects in a stream of data
//!
//! Top-K Heavykeeper algorithm for Top-K elephant flows
//!
//! This implementation is based on the paper HeavyKeeper: An Accurate Algorithm for Finding Top-k Elephant Flows
//! by Junzhi Gong, Tong Yang, Haowei Zhang, and Hao Li, Peking University; Steve Uhlig, Queen Mary, University of London;
//! Shigang Chen, University of Florida; Lorna Uden, Staffordshire University; Xiaoming Li, Peking University


mod heavykeeper;
mod priority_queue;

pub use self::heavykeeper::*;
