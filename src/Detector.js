import { useOpenCV } from './OpenCVProvider';
import { useCallback, useRef, useState } from 'react';
import styled from 'styled-components';
import * as tf from "@tensorflow/tfjs";


export default function Detector({model}) {
	const canvas = useRef(null);
	const canvas2 = useRef(null);
	const image = useRef(null);
	const cv = useOpenCV();
	
	const predict = useCallback(async (dom)=> {
		console.time("1");
		
		const w = 800, h = 360;
		const cvImage = cv.imread(dom.target, cv.IMREAD_COLOR);
		cv.cvtColor(cvImage, cvImage, cv.COLOR_RGBA2RGB);
		let img = tf.tensor(cvImage.data, [1, h, w, 3], 'float32'); 
	
		img = tf.sub(tf.div(img, 127.5), 1);
		img = model.predict(img);
		img = tf.greater(img, 0);
		img.data()
		.then((data)=>{
			img = cv.matFromArray(h, w, cv.CV_8UC1, data);
			let lines = new cv.Mat();
			let color = new cv.Scalar(255, 255, 0);

			cv.HoughLinesP(img, lines, 1, Math.PI/180, 190, 200, 30);
		
			for (let i = 0; i < lines.rows; ++i) {
				let startPoint = new cv.Point(
				  lines.data32S[i * 4],
				  lines.data32S[i * 4 + 1]
				);
				let endPoint = new cv.Point(
				  lines.data32S[i * 4 + 2],
				  lines.data32S[i * 4 + 3]
				);
				cv.line(cvImage, startPoint, endPoint, color);
			}
			cv.imshow(canvas.current, cvImage);
			img.delete(); lines.delete(); 

			console.timeEnd("1");
		});
	}, [cv, model])

	return (
		<>
			<Img ref={image} src={process.env.PUBLIC_URL + "/test1.jpg"} onLoad={predict} /> 
			<Canvas ref = { canvas }/> 
			<Canvas ref = { canvas2 }/> 
		</>
	)
}

const Img = styled.img `
	max-width: 800px;
`

const Canvas = styled.canvas `
	max-width: 800px;
	max-height: 800px;
`