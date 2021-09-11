import { useCallback, useEffect, useState } from 'react';
import OpenCVProvider from './OpenCVProvider';
import styled from 'styled-components'
import * as tf from "@tensorflow/tfjs";
import './ResizingLayer';
import Detector from './Detector';

function App() {
	const [model, setModel] = useState(null);
	const [isCVLoaded, setCVLoaded] = useState(false);
	
	// Load tf model
	useEffect(() => {
		tf.ready()
		.then(async () => {
			let tfmodel = await tf.loadLayersModel(process.env.PUBLIC_URL + "/edge_detector_MobileNetV2_0/model.json");
			setModel(tfmodel);
		})
	},[])

	// Load OpenCV
	const onLoadOpenCV = useCallback(()=>{
		setCVLoaded(true)
	},[])

	return (
		<AppContainer>
			<OpenCVProvider path={process.env.PUBLIC_URL + '/lib/opencv_improc.js'} onLoad={onLoadOpenCV} >
				{ model && isCVLoaded && <Detector model={model}/> }
			</OpenCVProvider>
		</AppContainer>
	);
}


const AppContainer = styled.div `
	width: 100%;
	height: 100%;
`

export default App;