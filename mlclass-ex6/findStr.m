function [idx] = findStr(vocabList, str)
	idx = -1;
	l = length(vocabList);
	for i = 1:l
  		if (strcmp(vocabList{i},str) == 1)
			idx = i;
			return;
		end
	end
end
