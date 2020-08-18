import 'dart:async';
import 'package:video_player/video_player.dart';
import 'package:flutter/material.dart';
import 'package:flutter/cupertino.dart';
import 'package:url_launcher/url_launcher.dart';

launchURL(String url) async
{
  if (await canLaunch(url))
    await launch(url);
  else
    throw 'Could not launch $url';
}

Widget makeButton( String caption, [String url] )
{
    if (url == null)
        return FlatButton
        (
            color: Colors.blue,
            highlightColor: Colors.blue[700],
            colorBrightness: Brightness.dark,
            splashColor: Colors.grey,
            child: Text(caption),
            shape:RoundedRectangleBorder(borderRadius: BorderRadius.circular(10.0)),
            onPressed: () {},
        );
    return FlatButton
    (
        color: Colors.blue[600],
        highlightColor: Colors.blue[900],
        colorBrightness: Brightness.dark,
        splashColor: Colors.grey,
        child: Text(caption),
        shape:RoundedRectangleBorder(borderRadius: BorderRadius.circular(10.0)),
        onPressed: () { launchURL(url); },
    );
}

class VideoPlayerScreen extends StatefulWidget
{
    VideoPlayerScreen({Key key, this.caption, this.url}) : super( key:key);
    final String caption;
    final String url;

    @override
    _VideoPlayerScreenState createState() => _VideoPlayerScreenState();
}

class _VideoPlayerScreenState extends State<VideoPlayerScreen>
{
    VideoPlayerController _controller;
    Future<void> _initializeVideoPlayerFuture;
    @override
    void initState()
    {

        _controller = VideoPlayerController.network( widget.url );
        _initializeVideoPlayerFuture = _controller.initialize();
        _controller.setLooping(true);
        super.initState();
    }

    @override
    void dispose()
    {
        _controller.dispose();
        super.dispose();
    }

    @override
    Widget build(BuildContext context)
    {
        return Column
        (
            children: <Widget>
            [
                FutureBuilder
                (
                    future: _initializeVideoPlayerFuture,
                    builder: (context, snapshot)
                    {
                      if (snapshot.connectionState == ConnectionState.done)
                          return AspectRatio( aspectRatio: _controller.value.aspectRatio, child: VideoPlayer(_controller),);
                      return Center(child: CircularProgressIndicator());
                    },
                ),
                Row
                (
                    children: <Widget>
                    [
                        Spacer(),
                        Text( widget.caption ),
                        Spacer(),
                        Text( 'Click to play/pause ⟿' ),
                        IconButton
                        (
                            onPressed: () { setState(() { if (_controller.value.isPlaying) { _controller.pause(); } else { _controller.play(); } }); },
                            icon: Icon( _controller.value.isPlaying ? Icons.pause : Icons.play_arrow,),
                            color: Colors.blue,
                        ),
                    ],
                ),
            ],
        );
    }
}


void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Multi-resolution convolutional neural networks for inverse problems',

      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: MyHomePage(title: 'Multi-resolution convolutional neural networks for inverse problems'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);
  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {


  @override
  Widget build(BuildContext context) {


    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title, style: TextStyle(fontWeight: FontWeight.bold),),
        centerTitle: true,
        primary: true,
      ),


	body:ListView
	(
		children:
		[
            Container // authors
            (
                height: 56.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                decoration: BoxDecoration(color: Colors.blue[300]),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Feng Wang[1,2]'),
                        Spacer(),
                        Text('Alberto Eljarrat[2]'),
                        Spacer(),
                        Text('Johannes Müller[2]'),
                        Spacer(),
                        Text('Trond R Henninen[2]'),
                        Spacer(),
                        Text('Rolf Erni[2]'),
                        Spacer(),
                        Text('Christoph T. Koch[1]'),
                        Spacer(),
                    ],
                ),
            ),
            Container // affiliations. if overflow in preview, create more containers
            (
                height: 56.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                decoration: BoxDecoration(color: Colors.blue[300]),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('[1] Electron Microscopy Center, Swiss Federal Laboratories for Materials Science and Technology, Überland Str. 129, CH-8600 Dübendorf, Switzerland'),
                        Spacer(),
                        Text('[2] Institut für Physik, IRIS Adlershof der Humboldt-Universität zu Berlin, 12489 Berlin, Germany'),
                        Spacer(),
                    ],
                ),
            ),

            Container // demo image
            (
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Image.asset('assets/images/MDCNN_6.jpg'),
            ),

            Container // demo image caption
            (
                height: 60.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('By matching outputs at all frequencies, MCNN achieves better stability and faster convergence than U-Net.', maxLines:2, style: TextStyle(fontWeight: FontWeight.bold)),
                        Spacer(),
                    ],
                ),
            ),


            Container
            (
                decoration: BoxDecoration(color: Colors.blue[300]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                    child: Text('Inverse problems in image processing, phase imaging, and computer vision often share the same structure of mapping input image(s) to output image(s) but are usually solved by different application-specific algorithms. Deep convolutional neural networks have shown great potential for highly variable tasks across many image-based domains, but sometimes can be challenging to train due to their internal non-linearity. We propose a novel, fast-converging neural network ar- chitecture capable of solving generic image(s)-to-image(s) inverse problems relevant to a diverse set of domains. We show this approach is useful in recovering wavefronts from direct intensity measurements, imaging objects from diffusely reflected images, and denoising scanning trans- mission electron microscopy images, just by using different training datasets. These successful applications demonstrate the proposed network to be an ideal candidate solving general inverse problems falling into the category of image(s)-to-image(s) translation.'),
                ),
            ),



            Container
            (
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                     child: VideoPlayerScreen
                     (
                        url: './assets/assets/movies/Extended.Video.1.diffusive.reflection.reconstruction.frame.1-256.mkv',
                        caption: 'The prediction from camera captured diffuse reflection images in the test set, with the first 256 frames displaying at 2 fps.',
                     ),
                ),
            ),

            Container
            (
                decoration: BoxDecoration(color: Colors.blue[100]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                     child: VideoPlayerScreen
                     (
                        url: './assets/assets/movies/Extended.Video.2.mcnn.denosing.512x512.mov',
                        caption: 'Denoising an atomic cluster that is rotating and reconstructing.',
                     ),
                ),
            ),

            Container
            (
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Center
                (
                     child: VideoPlayerScreen
                     (
                        url: './assets/assets/movies/Extended.Video.3.pgure-svt.mcnn.denoising.benchmark.128x128.mov',
                        caption: 'STEM images v.s. PGURE-SVT denoising v.s. MCNN denoising.',
                     ),
                ),
            ),

            // More images here

            Container //image asset
            (
                decoration: BoxDecoration(color: Colors.blue[100]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Image.asset('assets/images/piah_9th.jpg'),
            ),
            Container //image caption
            (
                decoration: BoxDecoration(color: Colors.blue[100]),
                height: 60.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Amplitudes and phases predicted from defocused images.', maxLines:2, style: TextStyle(fontWeight: FontWeight.bold)),
                        Spacer(),
                    ],
                ),
            ),

            Container //image asset
            (
                decoration: BoxDecoration(color: Colors.blue[50]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Image.asset('assets/images/tie_8th.jpg'),
            ),
            Container //image caption
            (
                decoration: BoxDecoration(color: Colors.blue[50]),
                height: 60.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Phase predicted from defocused image(s).', maxLines:2, style: TextStyle(fontWeight: FontWeight.bold)),
                        Spacer(),
                    ],
                ),
            ),

            Container //image asset
            (
                decoration: BoxDecoration(color: Colors.blue[100]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Image.asset('assets/images/denoising.jpg'),
            ),
            Container //image caption
            (
                decoration: BoxDecoration(color: Colors.blue[100]),
                height: 60.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Validation of the denoising application on heavily-noised aberration-corrected high-annular dark- field (HAADF) STEM images of sub-nanometre sized Platinum clusters.', maxLines:2, style: TextStyle(fontWeight: FontWeight.bold)),
                        Spacer(),
                    ],
                ),
            ),


            Container //image asset
            (
                decoration: BoxDecoration(color: Colors.blue[50]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Image.asset('assets/images/door_mirror_5th.jpg'),
            ),
            Container //image caption
            (
                decoration: BoxDecoration(color: Colors.blue[50]),
                height: 60.0, // in logical pixels
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        Text('Imaging objects from diffuse reflections.', maxLines:2, style: TextStyle(fontWeight: FontWeight.bold)),
                        Spacer(),
                    ],
                ),
            ),




            Container // buttons for short-cuts
            (
                decoration: BoxDecoration(color: Colors.blue[100]),
                padding: EdgeInsets.symmetric(horizontal:  MediaQuery.of(context).size.width * 0.1, vertical: 20.0),
                alignment: Alignment.center,
                child: Row
                (
                    children:<Widget>
                    [
                        Spacer(),
                        makeButton( 'Paper', 'https://www.nature.com/articles/s41598-020-62484-z' ),
                        Spacer(),
                        makeButton( 'Code', 'https://github.com/fengwang/MDCNN' ),
                        Spacer(),
                        makeButton( 'Online Demo', 'https://codeocean.com/capsule/6012862/tree/v1' ),
                        Spacer(),
                    ],
                ),
            ),

        ],
    ),

    );
  }
}

