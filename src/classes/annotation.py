import os
from typing import List, Tuple
import pandas as pd


class Annotation:
    """ Class to store seizure annotations as read in the tsv annotation files from the SeizeIT2 BIDS dataset.
    """
    def __init__(
        self,
        events: List[Tuple[int, int]],
        type: List[str],
        lateralization: List[str],
        localization: List[str],
        vigilance: List[str],
        rec_duration: float
    ):
        """Initiate an annotation instance

        Args:
            events (List([int, int])): list of tuples where each element contains the start and stop times in seconds of the event
            type (List[str]): list of event types according to the dataset's events dictionary (events.json).
            lateralization (List[str]): list of lateralization characteristics of the events according to the dataset's events dictionary (events.json).
            localization (List[str]): list of localization characteristics of the events according to the dataset's events dictionary (events.json).
            vigilance (List[str]): list of vigilance characteristics of the events according to the dataset's events dictionary (events.json).

        Returns:
            Annotation: returns an Annotation instance containing the events of the recording.
        """
        self.events = events
        self.types = type
        self.lateralization = lateralization
        self.localization = localization
        self.vigilance = vigilance
        self.rec_duration = rec_duration

    @classmethod
    def loadAnnotation(
        cls,
        annotation_path: str,
        recording: List[str],
    ):
        szEvents = list()
        szTypes = list()
        szLat = list()
        szLoc = list()
        szVig = list()

        tsvFile = os.path.join(annotation_path, recording[0], 'ses-01', 'eeg', '_'.join([recording[0], 'ses-01', 'task-szMonitoring', recording[1], 'events' + '.tsv']))
        df = pd.read_csv(tsvFile, delimiter="\t")
        for i, e in df.iterrows():
            if e['eventType'] != 'bckg' and e['eventType'] != 'impd':
                szEvents.append([e['onset'], e['onset'] + e['duration']])
                szTypes.append(e['eventType'])
                szLat.append(e['lateralization'])
                szLoc.append(e['localization'])
                szVig.append(e['vigilance'])
        durs = e['recordingDuration']

        return cls(
            szEvents,
            szTypes,
            szLat,
            szLoc,
            szVig,
            durs,
        )
