//使用 localStorage 的便捷hook
import { useState } from "react";

/**
 * @description useLocalStorage 将数据存于浏览器内，不受页面刷新影响
 * @param {string} key - 数据键值
 * @param {object} initialValue - 数据初始值
 * @param {boolean} isClearHistory - 是否清除旧数据
 * @return {[data:object, storeData:Function]}
 */
export function useLocalStorage(key, initialValue, isClearHistory) {

    //若isClear为真清除旧数据，完全删除键值
    const [data, setData] = useState(() => {
        isClearHistory && window.localStorage.removeItem(key);

        try {
            const item = window.localStorage.getItem(key);  //local storage 存储序列化形式的对象 (json string)
            if(item) {
                //localStorage 有数据的时候
                return JSON.parse(item);
            }
            else {
                //localStorage 上没有数据的时候，创建新的记录，使用初始默认数据
                window.localStorage.setItem(key, JSON.stringify(initialValue));
                return initialValue;
            }
        } 
        catch (error) {
            console.error(error);
            return initialValue;
        }
    });
  
    
    const storeData = (value) => {
        try {
            value = (value instanceof Function) ? value(data) : value; //value 也可以是函数与setState用法一样
            window.localStorage.setItem(key, JSON.stringify(value)); //先将对象序列化再存储
            setData(value); //程序内更新值
        } 
        catch (error) {
            console.error(error);
        }
    };
  
    return [data, storeData];
}