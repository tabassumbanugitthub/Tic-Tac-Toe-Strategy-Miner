# Tic-Tac-Toe Strategy Miner

A data-mining project built on tic-tac-toe. It generates every legal game, mines the data for patterns, builds a strategy from the statistics, and checks that strategy against perfect play. Built with **Python** and **Tkinter**, using only the standard library.

## How it works

1. **Generate:** every legal game (255,168 games, 5,478 positions, 958 final boards)
2. **Mine:** outcome statistics, opening heat-maps, and Apriori association rules
3. **Strategy:** score every move by its win and loss record over all continuations
4. **Validate:** compare the mined strategies with minimax in a tournament

## Features

- **Play:** play against three computer levels
  - Rookie: random moves
  - Miner: mined win-rate strategy
  - Master: mined statistics plus minimax-verified safety (it never loses)
- **Insights:** opening heat-maps and outcome statistics
- **Patterns:** association rules mined with Apriori, with CSV export
- **Arena:** tournament between the levels (100, 300, or 1000 games per matchup)

## Requirements

- Python 3.8 or newer
- Tkinter (included with the standard Python installers on Windows and macOS; on some Linux systems install it with `sudo apt install python3-tk`)

No third-party packages are needed.

## How to Run

```bash
git clone https://github.com/tabassumbanugitthub/Tic-Tac-Toe-Strategy-Miner.git
cd Tic-Tac-Toe-Strategy-Miner
python ttt_strategy_miner.py
```

## Screenshots

*Add screenshots of the Play page, the heat-map, and the Arena results here.*

## Project Structure

```
Tic-Tac-Toe-Strategy-Miner/
├── ttt_strategy_miner.py   # Mining engine and GUI
├── README.md
├── LICENSE
└── .gitignore
```

## Future Improvements

- Split the mining engine and GUI into separate modules
- Add unit tests for the game logic
- Extend the same pipeline to Connect Four

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
