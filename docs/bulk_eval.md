# Bulk Evaluation: 200 Scenes

**Date**: 2026-04-09
**Model**: gpt2-seinfeld int8 ONNX
**Pipeline**: with_chars prompt + selective injection + full postprocessing

## Summary

| Metric | Value |
|--------|-------|
| Total scenes | 198 |
| Good | 197 (99%) |
| Okay | 0 (0%) |
| Bad | 1 (1%) |
| Avg score | 30.1 |
| Avg characters | 3.2 |
| Avg dialogue turns | 14.0 |
| Avg unique ratio | 0.96 |
| Avg topic relevance | 0.43 |
| Avg word count | 100 |
| Has location tag | 92% |
| Has garbage chars | 0% |
| Avg gen time | 1.9s |

## Issue Breakdown

| Issue | Count | % |
|-------|-------|---|
| Single character only | 0 | 0% |
| Repetitive (unique < 0.5) | 0 | 0% |
| No dialogue turns | 0 | 0% |
| Off-topic (0 relevance) | 62 | 31% |
| Fragment lines | 110 | 56% |
| Gibberish/non-English | 1 | 1% |

## Score Distribution

| Range | Count | % |
|-------|-------|---|
| < 10 | 0 | 0% |
| 10-20 | 0 | 0% |
| 20-30 | 107 | 54% |
| 30-40 | 89 | 45% |
| > 40 | 2 | 1% |

## Sample Good Outputs

### a misunderstood invitation (score=41.7)
```
[GEORGE'S APARTMENT]

GEORGE: I thought you were invited. GEORGE: You were not. GEORGE: We're still friends. GEORGE: I'm just a little nervous. GEORGE: Don't be. GEORGE: I was wondering, what's your problem? GEORGE: I'm just wondering. GEORGE: Well, what if we cancel the meeting and just say we were just curious? GEORGE: All right, fine. You want to come, we'll just go to dinner. GEORGE: All right, let's just say we're just curious. GEORGE: Well, maybe we'll meet again. GEORGE: All right, what d
```

### a bad toupee (score=40.0)
```
[GEORGE'S APARTMENT]

GEORGE: Hey, hey, hey, hey. GEORGE: What's that? GEORGE: What? GEORGE: I thought you liked my toupee, Jerry. GEORGE: I thought I did. GEORGE: Well, it just... didn't quite come together right. GEORGE: I know. GEORGE: No, no, no. You know, I can't. It was a bad toupee. The only reason it came out like that was that I wasn't paying attention. GEORGE: Well, I'm not paying attention now. GEORGE: I'm listening. I'm paying attention. GEORGE: You know, I've always had this weird


```

### a misunderstood email (score=39.5)
```
[GEORGE'S CAR]

GEORGE: I'm on my way. GEORGE: Excuse me? GEORGE: The exclamation point. I was just wondering if I could use your car service. GEORGE: No, the point is I'm on my way. GEORGE: Okay, thanks. GEORGE: What?! GEORGE: I thought you meant "car service."

GEORGE: I didn't know you meant "car". You're a genius. Now get off of my car. GEORGE: I'm not going to do that!! GEORGE: I'm not going to do that!! I don't know why you're doing this!! I don't know what it is, I'm not going

JERRY: ?? 
```

### a terrible secret santa gift (score=39.4)
```
[MONK'S DINER]

JERRY: What is with all the presents? GEORGE: It's Christmas. JERRY: So you're happy now? GEORGE: I am. JERRY: You were not happy with the present. GEORGE: No, I wasn't happy with the present. JERRY: You were not happy with the present! GEORGE: I don't know what to tell you. You know that. JERRY: I told you not to tell me. You know that. GEORGE: Me too. JERRY: So what do you care? GEORGE: I'm not a bad person. JERRY: You're a bad person. All right. Let's just end

ELAINE: ?? GEOR
```

### a stain on a new shirt (score=39.0)
```
[JERRY'S APARTMENT]

JERRY: Hey George, what's this? GEORGE: A stain on the shirt. JERRY: A new shirt? GEORGE: Yeah. JERRY: Oh boy. GEORGE: What are you doing? JERRY: Well, it's a new shirt. GEORGE: What happened to your old one? JERRY: I bought it from the same guy. GEORGE: What guy? JERRY: I think he's wearing the same toupee. GEORGE: What's that? JERRY: It's uh, green. GEORGE: Green? JERRY: You know, the shade of green you wear on the beach? ELAINE: ?? JERRY: So, you got a new toupee? ELAINE:
```

## Sample Bad Outputs

### being stood up (score=25.7)
```
[MONK'S DINER]

JERRY: I'm standing up. JERRY: I'm fine. JERRY: I'm fine. Oh, thank you. Thanks. GEORGE: I don't trust this guy. JERRY: You're a good friend. GEORGE: He's a bad-man friend. JERRY: He's a bad-man-friend. GEORGE: He's a good-man-friend. JERRY: A bad-man-friend, and a good-man-friend. JERRY: I

ELAINE: 한국어요? JERRY: Hi. ELAINE: So I was standing and the guy behind me starts talking and I noticed he's got a big nose. JERRY: A big nose? What is it? ELAINE: It's big. Like a big hog's!
```

