import React, { createContext, useContext, useEffect, useState } from "react";


// ====== 自定义用户验证 context 对象 =============
const cvContext = createContext(null);

//========= functions ================
async function loadOpenCV(paths) {
    return new Promise((resolve, reject)=>{
        if(document.getElementById('OpenCV')){
            reject()
            return
        }    
        
        let script = document.createElement('script');
        script.setAttribute('id', 'OpenCV');
        script.setAttribute('async', '');
        script.setAttribute('type', 'text/javascript');
        script.addEventListener('load', () => {
            /* global cv */
            cv.then((cv)=>{
                resolve(cv)
            })
            .catch((e)=>{
                reject(e)
            })
        });
        script.addEventListener('error', () => {
            console.log('Failed to load opencv.js');
        });
        script.src = paths;
        let node = document.getElementsByTagName('script')[0];
        if (node.src !== paths) {
            node.parentNode.insertBefore(script, node);
        }
    })
}

// ====== 自定义context container =============
export default function OpenCVProvider({ children, path, onLoad }) {
    const [cv, setCV] = useState(null);

    useEffect(()=>{
        loadOpenCV(path)
        .then((loaded_cv)=>{
            setCV(loaded_cv)
            onLoad(loaded_cv)
        })
        .catch(()=>{
            console.log("OpenCV is already loaded")
        })
    },[path, onLoad]) 
    // eslint-disable-line react-hooks/exhaustive-deps

    const contextValue = cv;
    return (
        <cvContext.Provider value={contextValue}> 
            {children}
        </cvContext.Provider>
    )
} 

// ====== 自定义context hook =============
/**
 * @description OpenCVProvider 的子组件可以使用 useOpenCV() 获得cv api
 * @return {{cv:Object}}
 */
export function useOpenCV() {
    return useContext(cvContext);
}



