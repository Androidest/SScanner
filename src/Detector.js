import { useOpenCV } from './OpenCVProvider';
import { useCallback, useRef, useState } from 'react';
import styled from 'styled-components';
import * as tf from "@tensorflow/tfjs";

const pairThres = 50;
const concatThres = 5;

function sub(p1, p2) {
    return { x:p1.x-p2.x, y:p1.y-p2.y };
}

function isPair(p, index) {
    const p1 = p[0];
    const p2 = p[index];
    const dx = p1.x-p2.x;
    const dy = p1.y-p2.y;
    const dist =Math.sqrt( dx*dx + dy*dy)
    if ( dist < pairThres){
        p.splice(index, 1);
        p.splice(0, 1);
        return {x: (p1.x+p2.x)/2, y: (p1.y+p2.y)/2}
    }
    return null;
}

function concatLineSegments(cv, lineSeg) {
    const lines = [];
    const line_data = lineSeg.data32S;

    //compute mean lines from line segments
    for (let i = 0; i < lineSeg.rows; ++i) {
        const j = i * 4;
        const x1 = line_data[j], y1 = line_data[j + 1];
        const x2 = line_data[j + 2], y2 = line_data[j + 3];
        const theta = Math.atan2(x1-x2, y2-y1);
        const rho = x1*Math.cos(theta) + y1*Math.sin(theta);

        //concatinate to existing line
        let isInGroup = false;
        for (let l of lines) {
            const { sum_rho, sum_theta, count } = l;
            if (Math.abs(sum_rho/count - rho) < concatThres && Math.abs(sum_theta/count - theta) < concatThres) {
                l.sum_rho += rho;
                l.sum_theta += theta;
                l.x_min = Math.min(l.x_min, x1, x2);
                l.x_max = Math.max(l.x_max, x1, x2);
                l.y_min = Math.min(l.y_min, y1, y2);
                l.y_max = Math.max(l.y_max, y1, y2);
                ++l.count;
                isInGroup = true;
                break;
            }
        }

        // if no similar line exists, create new line
        if (!isInGroup) {
            const x = (x1 < x2)? {x_min:x1, x_max:x2} : {x_max:x1, x_min:x2};
            const y = (y1 < y2)? {y_min:y1, y_max:y2} : {y_max:y1, y_min:y2};
            lines.push({ sum_rho: rho, sum_theta:theta, count:1, ...x, ...y });
        }
    }

    // crop mean straight lines into segments
    const finalLines = [];
    for (let l of lines) {
        const { sum_rho, sum_theta, count, x_min, x_max, y_min, y_max} = l;
        const rho = sum_rho/count;
        const theta = sum_theta/count;
        const c = Math.cos(theta);
        const s = Math.sin(theta);

        const p = new Array(4);
        p[0] = { x:x_min, y:(rho-x_min*c)/s };
        p[1] = { x:x_max ,y:(rho-x_max*c)/s };
        p[2] = { x:(rho-y_min*s)/c, y:y_min };
        p[3] = { x:(rho-y_max*s)/c, y:y_max };
        
        let start  = isPair(p, 1) || isPair(p, 2) || isPair(p, 3);
        let end  = isPair(p, 1);
        if (start && end){
            finalLines.push({ start, end });
        }
    }
    return finalLines;
}

function intersection(l1, l2) {
    const {start:s1, end:e1} = l1;
    const {start:s2, end:e2} = l2;

    const s = sub(s2, s1);
    const d1 = sub(e1, s1);
    const d2 = sub(e2, s2);

    const cross = d1.x*d2.y - d1.y*d2.x;
    if (Math.abs(cross) < 1e-8)
        return null;

    const t1 = (s.x * d2.y - s.y * d2.x)/cross;
    const t2 = (s.x * d1.y - s.y * d1.x)/cross;
    if ( t1<0 || 1<t1 || t2<0 || 1<t2) 
        return null;

    return { x: s1.x+d1.x*t1 , y: s1.y+d1.y*t1 };
}


export default function Detector({model}) {
	const canvas = useRef(null);
	const canvas2 = useRef(null);
	const cv = useOpenCV();
    const mats = useRef({ 
        lineSeg: new cv.Mat(),
        color: new cv.Scalar(255, 255, 0)
    })
	
	const predict = useCallback(async (dom)=> {
		console.time("1");
        
		const w = 800, h = 360;
        const color = mats.current.color;
        const lineSeg = mats.current.lineSeg;
		const cvImage = cv.imread(dom.target, cv.IMREAD_COLOR);
		cv.cvtColor(cvImage, cvImage, cv.COLOR_RGBA2RGB);

		let img = tf.tensor(cvImage.data, [1, h, w, 3], 'float32'); 
		img = tf.sub(tf.div(img, 127.5), 1);
		img = model.predict(img);
		img = tf.greater(img, 0);
        tf.browser.toPixels(img.mul(1).reshape([h,w,1]), canvas2.current); // For Testing

		img.data()
		.then((data)=>{
			img = cv.matFromArray(h, w, cv.CV_8UC1, data);  // For Testing
			cv.HoughLinesP(img, lineSeg, 1, Math.PI/180, 140, 150, 50);

            for (let i = 0; i < lineSeg.rows; ++i) {
                let p1 = new cv.Point(lineSeg.data32S[i * 4], lineSeg.data32S[i * 4 + 1]);
                let p2 = new cv.Point(lineSeg.data32S[i * 4 + 2], lineSeg.data32S[i * 4 + 3]);
                cv.line(cvImage, p1, p2, [255,0,0,255])
            }

            const lines = concatLineSegments(cv, lineSeg);
            
            for (let i=0; i<lines.length-1; ++i) {
                const l1 = lines[i];
                for (let j=i+1; j<lines.length; ++j) {
                    const l2 = lines[j];
                    const p = intersection(l1, l2);
                    if (p) {
                        cv.circle(cvImage, p, 2, color, 2);
                    }
                }
                const { start, end } = l1;
                cv.line(cvImage, start, end, color);
            } 
			cv.imshow(canvas.current, cvImage);
			img.delete();
            cvImage.delete();

			console.timeEnd("1");
		});
	}, [cv, model])

	return (
		<>
			<Img src={process.env.PUBLIC_URL + "/test2.jpg"} onLoad={predict} /> 
			<Canvas ref = { canvas2 }/> 
			<Canvas ref = { canvas }/> 
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