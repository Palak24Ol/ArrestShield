# Raw Dataset Profile

Generated from immutable source files. Counts do not imply that records are ready for training.

## indian_cyber_scam_phonecall_hinglish

```json
{
  "rows": 10000,
  "columns": [
    "text",
    "label",
    "scam_category",
    "caller_type",
    "audio_duration",
    "urgency_level",
    "contains_blackmail",
    "language_style"
  ],
  "encoding": "utf-8-sig",
  "categorical_counts": {
    "label": {
      "0": 5000,
      "1": 5000
    },
    "scam_category": {
      "none": 5000,
      "police_digital_arrest": 1776,
      "police_blackmail": 1460,
      "bank_kyc": 718,
      "amazon": 273,
      "aadhaar": 260,
      "lottery": 259,
      "relative": 254
    },
    "caller_type": {
      "police_impersonator": 3236,
      "delivery_agent": 1052,
      "other_scammer": 1046,
      "doctor_clinic": 1029,
      "family": 999,
      "friend": 982,
      "office_colleague": 938,
      "bank_scammer": 718
    },
    "urgency_level": {
      "low": 5000,
      "medium": 2676,
      "high": 2324
    },
    "contains_blackmail": {
      "False": 8540,
      "True": 1460
    },
    "language_style": {
      "hinglish": 10000
    }
  },
  "unique_counts": {
    "text": 743,
    "label": 2,
    "scam_category": 8,
    "caller_type": 8,
    "audio_duration": 246,
    "urgency_level": 3,
    "contains_blackmail": 2,
    "language_style": 1
  }
}
```

## indian_multilingual_scam_messages

```json
{
  "rows": 120,
  "columns": [
    "message",
    "label",
    "reason",
    "domain",
    "language"
  ],
  "encoding": "utf-8-sig",
  "categorical_counts": {
    "label": {
      "scam": 60,
      "legit": 60
    },
    "reason": {
      "Urgency, reward or suspicious instruction": 60,
      "Normal transactional message": 60
    },
    "domain": {
      "ecommerce": 27,
      "government": 24,
      "banking": 22,
      "telecom": 19,
      "finance": 19,
      "utilities": 9
    },
    "language": {
      "Hinglish": 42,
      "English": 41,
      "Hindi": 37
    }
  },
  "unique_counts": {
    "message": 49,
    "label": 2,
    "reason": 2,
    "domain": 6,
    "language": 3
  }
}
```

## synthetic_scam_dialogue

```json
{
  "rows": 1600,
  "columns": [
    "dialogue",
    "type",
    "label"
  ],
  "encoding": "utf-8-sig",
  "categorical_counts": {
    "type": {
      "ssn": 200,
      "refund": 200,
      "support": 200,
      "reward": 200,
      "delivery": 200,
      "insurance": 200,
      "telemarketing": 200,
      "wrong": 200
    },
    "label": {
      "1": 800,
      "0": 800
    }
  },
  "unique_counts": {
    "dialogue": 1598,
    "type": 8,
    "label": 2
  }
}
```

## synthetic_multi_agent_scam_conversation

```json
{
  "rows": 1600,
  "columns": [
    "dialogue",
    "personality",
    "type",
    "labels"
  ],
  "encoding": "utf-8-sig",
  "categorical_counts": {
    "personality": {
      "skeptical": 200,
      "distracted": 200,
      "polite": 200,
      "aggressive": 200,
      "trusting": 200,
      "confused": 200,
      "anxious": 200,
      "greedy": 200
    },
    "type": {
      "ssn": 200,
      "reward": 200,
      "refund": 200,
      "support": 200,
      "delivery": 200,
      "appointment": 200,
      "insurance": 200,
      "wrong": 200
    },
    "labels": {
      "1": 800,
      "0": 800
    }
  },
  "unique_counts": {
    "dialogue": 1600,
    "personality": 8,
    "type": 8,
    "labels": 2
  }
}
```

## banking77_train

```json
{
  "rows": 10003,
  "columns": [
    "text",
    "category"
  ],
  "encoding": "utf-8-sig",
  "categorical_counts": {},
  "unique_counts": {
    "text": 9999,
    "category": 77
  }
}
```

## banking77_test

```json
{
  "rows": 3080,
  "columns": [
    "text",
    "category"
  ],
  "encoding": "utf-8-sig",
  "categorical_counts": {},
  "unique_counts": {
    "text": 3079,
    "category": 77
  }
}
```

## daily_dialog

```json
{
  "archives": {
    "test.zip": {
      "files": [
        {
          "name": "test/dialogues_act_test.txt",
          "bytes": 16480
        },
        {
          "name": "test/dialogues_emotion_test.txt",
          "bytes": 16480
        },
        {
          "name": "test/dialogues_test.txt",
          "bytes": 544208
        }
      ],
      "dialogues": 1000
    },
    "train.zip": {
      "files": [
        {
          "name": "train/dialogues_act_train.txt",
          "bytes": 185458
        },
        {
          "name": "train/dialogues_emotion_train.txt",
          "bytes": 185458
        },
        {
          "name": "train/dialogues_train.txt",
          "bytes": 6041175
        }
      ],
      "dialogues": 11118
    },
    "validation.zip": {
      "files": [
        {
          "name": "validation/dialogues_act_validation.txt",
          "bytes": 17138
        },
        {
          "name": "validation/dialogues_emotion_validation.txt",
          "bytes": 17138
        },
        {
          "name": "validation/dialogues_validation.txt",
          "bytes": 558030
        }
      ],
      "dialogues": 1000
    }
  },
  "total_dialogues": 13118
}
```

## schema_guided_dialogue

```json
{
  "json_files": 181,
  "dialogues": 22825,
  "turns": 463284,
  "services": {
    "Travel_1": 2808,
    "Restaurants_1": 2419,
    "Events_2": 2385,
    "Flights_1": 2070,
    "Movies_1": 1926,
    "Weather_1": 1783,
    "Calendar_1": 1602,
    "Events_1": 1542,
    "Hotels_1": 1461,
    "Buses_1": 1398,
    "Hotels_3": 1345,
    "Hotels_2": 1279,
    "RideSharing_2": 1265,
    "RentalCars_1": 1249,
    "Buses_2": 1211,
    "Services_1": 1205,
    "Media_1": 1113,
    "Homes_1": 1027,
    "RideSharing_1": 958,
    "Hotels_4": 907,
    "Music_1": 884,
    "Restaurants_2": 799,
    "Banks_1": 727,
    "RentalCars_2": 717,
    "Services_3": 684,
    "Flights_2": 677,
    "Music_2": 602,
    "Events_3": 592,
    "RentalCars_3": 544,
    "Services_2": 534,
    "Services_4": 533,
    "Buses_3": 526,
    "Flights_4": 506,
    "Flights_3": 391,
    "Media_3": 364,
    "Trains_1": 350,
    "Music_3": 347,
    "Alarm_1": 324,
    "Messaging_1": 298,
    "Banks_2": 294,
    "Movies_3": 272,
    "Homes_2": 246,
    "Payment_1": 222,
    "Media_2": 179,
    "Movies_2": 141
  },
  "splits": {
    "dev": 2482,
    "test": 4201,
    "train": 16142
  }
}
```

## hinmix_hicmrom_test

```json
{
  "error": "pyarrow unavailable: No module named 'pyarrow'"
}
```

## hinmix_hicmrom_valid

```json
{
  "error": "pyarrow unavailable: No module named 'pyarrow'"
}
```

## hinmix_noisy_test

```json
{
  "error": "pyarrow unavailable: No module named 'pyarrow'"
}
```

## hinmix_noisy_valid

```json
{
  "error": "pyarrow unavailable: No module named 'pyarrow'"
}
```

## Interpretation

- Public scam corpora are mostly synthetic or weakly documented; they require deduplication and manual label audit.
- BANKING77, DailyDialog, and Schema-Guided Dialogue provide legitimate-domain hard negatives, but they are not phone-call recordings.
- HINMIX is language robustness data only and must not receive scam labels.
- The primary English/Hindi/Hinglish ArrestShield corpus with turn-level tactic/stage labels is still required.
