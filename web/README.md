# MCNN webpage

A Flutter project to present MCNN as a standalone webpage. Preview available at [__here__](http://fengwang.github.io/mdcnn/#/).

## Use this repo to generate your own paper webpage

0. copy the pictures and movies related to your paper to folder `assets/images/` and `assets/movies/`,
1. edit file `pubspec.yaml`, update the `description` part and the `assets` part.  The `assets` part is supposed to reflect all the media files just copied in the last step,
3. edit corresponding part of  file `lib/main.dart`, including
    - replace the `title` (line 118 and line 124) with your paper's title,
    - replace the `author` starting from line 165, (note the authors are seperated with a `Spacer()`)
    - replace the `affiliation` part starting from line 191,
    - replace the leading image (line 202) and its caption (line 214) of your paper, which should have been copied into the `assets/images` folder at the first step, (if your paper do not have a leading image, just comment out/delete the Containers from line 199 to line 218)
    - replace the `abstract` part (line 228),
    - replace the `url`s and `caption`s of containers reflecting the videos copied to the `assets/videos` folder at the first step, (from line 233 to line 275, feel free to add or delete)
    - replace the image `asset` and `caption` reflecting additional images copied at the first step, (from line 278 to line 366, feel free to add or delete)
    - replace the corresponding urls for the bottom link, if any, (delete the button or leave the link empty if not available)
4. install and config [flutter](https://flutter.dev/docs/get-started/web) to build a web application, (refer to the `Makefile` for compiliation details)
5. preview the generated web application and select a proper web sapce to host it. The Generated static website is located at folder `build/web`.

The sourcecode code of [Noise2Atom](https://github.com/fengwang/Noise2Atom/tree/master/noise2atom_webpage/webpage) for your reference. Please file an issue if you have any questions.

