//通用Style
import styled, { createGlobalStyle, css } from "styled-components"

//############### Styled #############################################################################
export const Styled = {
    
}



//############### CSS #############################################################################
const flex_row = css `
    display: flex;
    flex-direction: row;
`;

const flex_col = css `
    display: flex;
    flex-direction: column;
`;

export const Css = {
    //简化 flex的使用
    //==================== flex row =====================
    flex_row: {
        horz: {
            left : css ` ${flex_row} justify-content: flex-start;`,
            center : css ` ${flex_row} justify-content: center;`,
            right : css ` ${flex_row} justify-content: flex-end;`,
            in_space : css ` ${flex_row} justify-content: space-between;`,
            in_out_space : css ` ${flex_row} justify-content: space-evenly;`
        },
        
        vert: {
            top : css `align-items:flex-start;`,
            center : css `align-items:center;`,
            bottom : css `align-items:flex-end;`,
            stretch : css `align-items:stretch;`
        }
    },
    //简化 flex的使用
    //==================== flex col =====================
    flex_col: {
        vert: {
            top : css ` ${flex_col} justify-content: flex-start;`,
            center : css ` ${flex_col} justify-content: center;`,
            bottom : css ` ${flex_col} justify-content: flex-end;`,
            in_space : css ` ${flex_col} justify-content: space-between;`,
            in_out_space : css ` ${flex_col} justify-content: space-evenly;`
        },
        
        horz: {
            left : css `align-items:flex-start;`,
            center : css `align-items:center;`,
            right : css `align-items:flex-end;`,
            stretch : css `align-items:stretch;`
        }
    },

    //========== css transition 闪光动画 =============
    hover_blink: {
        //从往右的闪光效果
        forward: css `
            :after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 0;
                height: 100%;
                background-color: rgba(255,255,255,0.4);
                -webkit-transition: none;
                -moz-transition: none;
                transition: none;
                border-radius: 6ch;
            }
            
            :hover:after {
                width: 120%;
                background-color: rgba(255,255,255,0);
                -webkit-transition: all 1s ease-in-out;
                -moz-transition: all 1s ease-in-out;
                transition: all 1s ease-in-out;
            }
        `,
        //效果往回缩
        backward: css `
            :not(:hover):after {
            width: 0;
            background-color: rgba(255,255,255,0.4);
            -webkit-transition: all 0.5s ease-out;
            -moz-transition: all 0.5s ease-out;
            transition: all 0.5s ease-out;
        }`
    },

    img_protection: css `
        ${'' /* 防止用户鼠标拖动选择出现蓝筐 */}
        user-select: none;
        ${'' /* 防止拖动图片 */}
        user-drag: none;
        ${'' /* 防止下载图片 */}
        pointer-events: none;
    `
}



//############### Global Style #####################################################################
export const GlobalStyle = createGlobalStyle `
    html, body, #root {
        height: 100%;
        width: 100%;
	    overflow: hidden;
    }

    body {
        margin: 0;
        background-color: rgb(255, 255, 255);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
            'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
            sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        user-select: none; 
        user-drag: none;
    }

    .img {
        ${Css.img_protection}
    }

    code {
        font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New', monospace;
    }
`; 