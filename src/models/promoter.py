import shutil

class ChampionPromoter:
    def promote(self):
        shutil.copy(
            "data/models/candidate.pkl",
            "data/models/champion.pkl"
        )
